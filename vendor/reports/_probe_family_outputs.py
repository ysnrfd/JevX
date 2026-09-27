"""Probe 4: forward-contract verification for audited families (tiny random configs, CPU).

For each family: instantiate base model from a tiny config, run forward with synthetic inputs,
record output class name, non-None field names, and presence/shape of pooler_output /
encoder_last_hidden_state / hidden_states. Evidence for the compatibility matrix.
"""
import sys, json, warnings
sys.path.insert(0, r'C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\transformers\src')
import torch

warnings.filterwarnings('ignore')
results = {}

def rec(name, fn):
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = {'error': f'{type(e).__name__}: {str(e)[:300]}'}

def fields(out):
    d = {}
    for k in out.keys():
        v = out[k]
        if torch.is_tensor(v):
            d[k] = tuple(v.shape)
        elif isinstance(v, (tuple, list)):
            d[k] = f'tuple[{len(v)}]'
        else:
            d[k] = type(v).__name__
    return d

IDS = torch.randint(0, 100, (2, 7))
MASK = torch.ones(2, 7, dtype=torch.long)

# --- encoder-only ---
def bert():
    from transformers import BertConfig, BertModel
    m = BertModel(BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100))
    o = m(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('bert', bert)

def roberta():
    from transformers import RobertaConfig, RobertaModel
    c = RobertaConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100, max_position_embeddings=64)
    m = RobertaModel(c)
    o = m(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('roberta', roberta)

def deberta():
    from transformers import DebertaV2Config, DebertaV2Model
    c = DebertaV2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100, max_position_embeddings=64, relative_attention=False, position_biased_input=False)
    m = DebertaV2Model(c)
    o = m(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('deberta_v2', deberta)

def albert():
    from transformers import AlbertConfig, AlbertModel
    c = AlbertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100, max_position_embeddings=64, num_hidden_groups=1)
    m = AlbertModel(c)
    o = m(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('albert', albert)

def distilbert():
    from transformers import DistilBertConfig, DistilBertModel
    c = DistilBertConfig(dim=32, n_layers=2, n_heads=4, hidden_dim=64, vocab_size=100, max_position_embeddings=64)
    m = DistilBertModel(c)
    o = m(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('distilbert', distilbert)

# --- decoder-only ---
def dec(name, cfg_cls, kw, cache_flag=True):
    def run():
        c = cfg_cls(**kw)
        m = cfg_cls.__module__  # noqa
        import importlib
        model_mod = importlib.import_module('transformers')
        Model = getattr(model_mod, name)
        mm = Model(c)
        o = mm(input_ids=IDS, attention_mask=MASK, use_cache=False, output_hidden_states=True)
        d = {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
        # cache on/off check
        o2 = mm(input_ids=IDS, attention_mask=MASK, use_cache=True)
        d['use_cache_true_has_pkv'] = o2.past_key_values is not None
        d['pooler_attr'] = hasattr(mm, 'pooler') and mm.pooler is not None
        return d
    return run

rec('llama', dec('LlamaModel', __import__('transformers').LlamaConfig,
    dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100)))
rec('qwen2', dec('Qwen2Model', __import__('transformers').Qwen2Config,
    dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100)))
rec('mistral', dec('MistralModel', __import__('transformers').MistralConfig,
    dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100)))
rec('gemma', dec('GemmaModel', __import__('transformers').GemmaConfig,
    dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100)))
rec('gpt2', dec('GPT2Model', __import__('transformers').GPT2Config,
    dict(n_embd=32, n_layer=2, n_head=4, vocab_size=100, n_positions=64)))
def phi3():
    from transformers import Phi3Config, Phi3Model
    c = Phi3Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100, max_position_embeddings=64, pad_token_id=0)
    m = Phi3Model(c)
    o = m(input_ids=IDS, attention_mask=MASK, use_cache=False, output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states),
            'pooler_attr': hasattr(m, 'pooler') and m.pooler is not None}
rec('phi3', phi3)

# --- encoder-decoder ---
def t5():
    from transformers import T5Config, T5Model
    c = T5Config(vocab_size=100, d_model=32, d_ff=64, num_layers=2, num_heads=4, num_decoder_layers=1)
    m = T5Model(c); m.eval()
    dec_ids = torch.randint(0, 100, (2, 5))
    with torch.no_grad():
        o = m(input_ids=IDS, decoder_input_ids=dec_ids, output_hidden_states=True)
        enc = m.encoder(input_ids=IDS)
        direct_eq = torch.equal(enc.last_hidden_state, o.encoder_last_hidden_state)
    return {'cls': type(o).__name__, 'fields': fields(o),
            'enc_hs_len': len(o.encoder_hidden_states), 'encoder_direct_equals_field': direct_eq,
            'direct_encoder_cls': type(enc).__name__}
rec('t5', t5)

def bart():
    from transformers import BartConfig, BartModel
    c = BartConfig(vocab_size=100, d_model=32, encoder_layers=2, decoder_layers=1, encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64)
    m = BartModel(c); m.eval()
    with torch.no_grad():
        o = m(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
        enc = m.encoder(input_ids=IDS, attention_mask=MASK)
        direct_eq = torch.equal(enc.last_hidden_state, o.encoder_last_hidden_state)
    return {'cls': type(o).__name__, 'fields': fields(o),
            'enc_hs_len': len(o.encoder_hidden_states), 'encoder_direct_equals_field': direct_eq,
            'direct_encoder_cls': type(enc).__name__}
rec('bart', bart)

def whisper():
    from transformers import WhisperConfig, WhisperModel
    c = WhisperConfig(vocab_size=100, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=40)
    m = WhisperModel(c); m.eval()
    feats = torch.randn(2, 40, 64)
    with torch.no_grad():
        o = m(input_features=feats, output_hidden_states=True)
        enc = m.encoder(input_features=feats)
        direct_eq = torch.equal(enc.last_hidden_state, o.encoder_last_hidden_state)
    return {'cls': type(o).__name__, 'fields': fields(o),
            'encoder_direct_equals_field': direct_eq, 'direct_encoder_cls': type(enc).__name__}
rec('whisper', whisper)

# --- vision / multimodal ---
def vit():
    from transformers import ViTConfig, ViTModel
    c = ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, image_size=32, patch_size=8, num_channels=3)
    m = ViTModel(c)
    o = m(pixel_values=torch.randn(2, 3, 32, 32), output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('vit', vit)

def clip_text_vision():
    from transformers import CLIPTextModel, CLIPVisionModel, CLIPTextConfig, CLIPVisionConfig
    tc = CLIPTextConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100, max_position_embeddings=64)
    tm = CLIPTextModel(tc)
    to = tm(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    vc = CLIPVisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, image_size=32, patch_size=8)
    vm = CLIPVisionModel(vc)
    vo = vm(pixel_values=torch.randn(2, 3, 32, 32), output_hidden_states=True)
    return {'text_cls': type(to).__name__, 'text_fields': fields(to),
            'vision_cls': type(vo).__name__, 'vision_fields': fields(vo)}
rec('clip', clip_text_vision)

def siglip():
    from transformers import SiglipTextModel, SiglipVisionModel, SiglipTextConfig, SiglipVisionConfig
    tc = SiglipTextConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100, max_position_embeddings=64)
    tm = SiglipTextModel(tc)
    to = tm(input_ids=IDS, attention_mask=MASK, output_hidden_states=True)
    vc = SiglipVisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, image_size=32, patch_size=8)
    vm = SiglipVisionModel(vc)
    vo = vm(pixel_values=torch.randn(2, 3, 32, 32), output_hidden_states=True)
    return {'text_cls': type(to).__name__, 'text_fields': fields(to),
            'vision_cls': type(vo).__name__, 'vision_fields': fields(vo)}
rec('siglip', siglip)

def llava_next():
    from transformers import LlavaNextConfig, LlavaNextModel
    from transformers import CLIPVisionConfig, LlamaConfig
    vision = CLIPVisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, image_size=32, patch_size=8)
    text = LlamaConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100, pad_token_id=0)
    c = LlavaNextConfig(vision_config=vision, text_config=text)
    c.image_grid_pinpoints = [[32, 32]]
    m = LlavaNextModel(c)
    # input with an image placeholder token (id 32000 not in vocab 100; use a normal id + no image)
    ids = torch.randint(0, 100, (1, 6))
    o = m(input_ids=ids, attention_mask=torch.ones(1, 6, dtype=torch.long), use_cache=False)
    d = {'cls': type(o).__name__, 'fields': fields(o)}
    # with image: pixel_values shaped (num_images, C, H, W) + image_sizes
    try:
        pv = torch.randn(1, 3, 32, 32)
        o2 = m(input_ids=ids, pixel_values=pv, image_sizes=torch.tensor([[32, 32]]), attention_mask=torch.ones(1, 6, dtype=torch.long), use_cache=False)
        d['with_image_fields'] = fields(o2)
    except Exception as e:
        d['with_image_error'] = f'{type(e).__name__}: {str(e)[:200]}'
    return d
rec('llava_next', llava_next)

# --- audio ---
def wav2vec2():
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    c = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=32, conv_dim=(32, 32), conv_stride=(4, 4), conv_kernel=(4, 4), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, do_stable_layer_norm=False)
    m = Wav2Vec2Model(c)
    o = m(input_values=torch.randn(2, 160), output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('wav2vec2', wav2vec2)

def ast():
    from transformers import ASTConfig, ASTModel
    c = ASTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, num_mel_bins=16, max_num_frames=32, patch_size=(16, 16), window_size=16)
    m = ASTModel(c)
    o = m(input_values=torch.randn(2, 16, 40), output_hidden_states=True)
    return {'cls': type(o).__name__, 'fields': fields(o), 'hs_len': len(o.hidden_states)}
rec('ast', ast)

print(json.dumps(results, indent=2, default=str))
