"""Probe 3: wrapper holding a TIED model (LlamaForCausalLM) — Q1 composition + safetensors tie removal."""
import sys, os, json, tempfile
sys.path.insert(0, r'C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\transformers\src')
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel, PreTrainedConfig

results = {}

class WrapConfig(PreTrainedConfig):
    model_type = "jevx_wrap"
    def __init__(self, **kw):
        self.inner = kw.pop("inner", None)
        super().__init__(**kw)

class WrapModel(PreTrainedModel):
    config_class = WrapConfig
    def __init__(self, config):
        super().__init__(config)
        self.inner = LlamaForCausalLM(LlamaConfig.from_dict(config.inner))
        self.head = torch.nn.Linear(16, 8)
        self.post_init()
    def forward(self, input_ids=None, **k):
        return self.inner(input_ids=input_ids)

lc = LlamaConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=4,
                 num_key_value_heads=2, intermediate_size=32, vocab_size=50)
cfg = WrapConfig()
cfg.inner = lc.to_dict()
m = WrapModel(cfg)
ids = torch.randint(0, 50, (2, 5))
m(input_ids=ids)

# check tie state in wrapper
sd = m.state_dict()
results['wrapper_state_has_lm_head'] = 'inner.lm_head.weight' in sd
results['wrapper_state_has_embed'] = 'inner.model.embed_tokens.weight' in sd
if 'inner.lm_head.weight' in sd and 'inner.model.embed_tokens.weight' in sd:
    results['wrapper_tied_same_storage'] = sd['inner.lm_head.weight'].data_ptr() == sd['inner.model.embed_tokens.weight'].data_ptr()
results['wrapper_all_tied_keys'] = dict(getattr(m, 'all_tied_weights_keys', {}) or {})

with tempfile.TemporaryDirectory() as d:
    try:
        m.save_pretrained(d)
        results['tied_wrapper_save'] = 'ok'
        results['saved_files'] = sorted(os.listdir(d))
        m2 = WrapModel.from_pretrained(d)
        sd2 = m2.state_dict()
        keys1, keys2 = set(sd), set(sd2)
        results['key_sets_equal'] = keys1 == keys2
        # after reload, is lm_head tied again?
        if 'inner.lm_head.weight' in sd2:
            results['reloaded_lm_head_tied'] = sd2['inner.lm_head.weight'].data_ptr() == sd2['inner.model.embed_tokens.weight'].data_ptr()
        # value equality on shared keys
        common = keys1 & keys2
        results['weights_equal'] = all(torch.equal(sd[k].float(), sd2[k].float()) for k in common)
        # forward after reload works
        out = m2(input_ids=ids)
        results['reloaded_forward_ok'] = True
    except Exception as e:
        results['tied_wrapper_save'] = f"{type(e).__name__}: {str(e)[:400]}"

# dtype placement probe: wrapper built inside from_pretrained(dtype=bf16) — child+head both cast?
with tempfile.TemporaryDirectory() as d:
    m.save_pretrained(d)
    m3 = WrapModel.from_pretrained(d, dtype=torch.bfloat16)
    results['bf16_head'] = str(next(m3.head.parameters()).dtype)
    results['bf16_inner_embed'] = str(next(m3.inner.model.embed_tokens.parameters()).dtype)

print(json.dumps(results, indent=2, default=str))
