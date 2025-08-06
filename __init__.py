# Import custom nodes
from custom_nodes.enhancement_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Add our custom nodes
NODE_CLASS_MAPPINGS.update({
    'PerformanceOptimize': PerformanceOptimizeNode,
    'ImageQualityEnhance': ImageQualityEnhanceNode,
    'CreatePatchImage': CreatePatchImageNode,
})

NODE_DISPLAY_NAME_MAPPINGS.update({
    'PerformanceOptimize': 'Optimize Performance',
    'ImageQualityEnhance': 'Enhance Image Quality',
    'CreatePatchImage': 'Create Patch Image',
})