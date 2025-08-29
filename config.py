from transformers.configuration_utils import PretrainedConfig

from typing import Dict

class LoopFormerForImageClassificationConfig(PretrainedConfig):
    
    model_type = "loopformer_for_image_classification"
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_loops: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        
        # Module configuration
        modules: Dict[str, int] = None,
        
        # Router configuration
        router_type: str = "linear",
        router_act: str = "gelu",
        
        # Image processing
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.hidden_size = hidden_size
        self.num_loops = num_loops
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        
        if modules is None:
            self.modules = {
                "full_attention": 4,
                "mlp": 4,
                "swish_glu": 4,
                "identity": 1
            }
        else:
            self.modules = modules
        
        self.router_type = router_type
        self.router_act = router_act
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        
        