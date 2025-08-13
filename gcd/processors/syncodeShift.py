import torch

from transformers import LogitsProcessor


class BatchSyncodeCallable:
    """
    Adapt per-row SyncodeProcessor.process(prev_ids, logits) -> logits
    to a batch callable: (input_ids, scores) -> scores
    """

    def __init__(self, syncode_processor):
        self.proc = syncode_processor

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Expect input_ids: (B, L), scores: (B, V)
        B = input_ids.size(0)

        # Keep originals to restore dtype/device at the end
        orig_dtype = scores.dtype
        orig_device = scores.device

        # Work device: use scores' device by default
        work_device = orig_device

        outs = []
        for b in range(B):
            prev_ids_b = input_ids[b:b + 1].to(device=work_device, dtype=torch.long, non_blocking=True)  # (1, L)
            # Many custom processors assume float32 logits
            scores_b = scores[b:b + 1].to(device=work_device, dtype=torch.float32, non_blocking=True)  # (1, V)

            out_b = self.proc.process(prev_ids_b, scores_b)  # should return (1, V) float32
            # Be defensive about dtype/device coming back
            out_b = out_b.to(device=work_device, dtype=torch.float32, non_blocking=True)

            outs.append(out_b)

        out = torch.cat(outs, dim=0)  # (B, V)
        # Cast back to the dtype HF generation is using (e.g., float16 on GPU)
        out = out.to(device=orig_device, dtype=orig_dtype, non_blocking=True)
        return out


class SyncodeWithFProcessor(LogitsProcessor):
    def __init__(self, *, grammar_text, tokenizer, parser_state_extractor_cls, f_shift_fn,
                 syncode_proc=None, stack_context_length=3, decode_token_fn=None):
        self.grammar_text = grammar_text
        self.tokenizer = tokenizer
        self.ParserStateExtractorCls = parser_state_extractor_cls
        self.f_shift_fn = f_shift_fn
        self.syncode_proc = syncode_proc
        self.stack_context_length = stack_context_length
        self.decode_token_fn = decode_token_fn or (lambda tid: tokenizer.decode([tid], skip_special_tokens=True))

    def _build_extractor_for_prefix(self, prefix_text: str):
        ex = self.ParserStateExtractorCls(self.grammar_text)
        ex.advance_parser(prefix_text, top_k=self.stack_context_length, prefix_text=prefix_text)
        return ex

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Optionally apply Syncode first
        if self.syncode_proc is not None:
            scores = self.syncode_proc(input_ids, scores)

        B, V = scores.shape
        neginf_mask = torch.isneginf(scores)

        for b in range(B):
            prefix_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            print("prefix_text", prefix_text)
            # collect valid token ids once
            valid_ids = torch.nonzero(~neginf_mask[b], as_tuple=False).flatten().tolist()
            if not valid_ids:
                continue

            # For each candidate, NEW extractor → prefix → advance by token (no deepcopy!)
            for tid in valid_ids:
                tok_text = self.decode_token_fn(tid)

                ex_next = self._build_extractor_for_prefix(prefix_text)
                res = ex_next.advance_parser(tok_text, top_k=self.stack_context_length,
                                             prefix_text=prefix_text + tok_text)

                state_oh = res.get('onehot_current_state', [])
                stack_names = res.get('stack', [])
                remainder = getattr(ex_next, "current_remainder", "")

                delta = float(self.f_shift_fn(state_oh, stack_names, remainder))
                if delta != 0.0:
                    scores[b, tid] = scores[b, tid] + scores.new_tensor(delta)

        # Never revive disallowed tokens
        return torch.where(neginf_mask, scores.new_full(scores.shape, float("-inf")), scores)
