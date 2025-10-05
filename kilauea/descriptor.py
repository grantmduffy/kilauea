import vulkan as vk
import numpy as np


class DescriptorSet:

    def __init__(self, app, n_images=1):
        self.app = app
        self.objects = []
        self.n_images = n_images
        self._vk_layouts = []
        self._vk_descriptor_sets = []

    def add(self, item: Texture | UniformBuffer, stages=vk.VK_SHADER_STAGE_COMPUTE_BIT):
        binding = len(self.objects)
        self._vk_layouts.append(item.get_layout_binding(binding, stages))
        self.objects.append(item)

    def create(self, update=True):
        self._vk_descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            self.app._vk_device,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(self._vk_layouts),
                pBindings=self._vk_layouts
            ),
            None
        )
        self._vk_descriptor_sets = vk.vkAllocateDescriptorSets(
            self.app._vk_device,
            vk.VkDescriptorSetAllocateInfo(
                descriptorPool=self.app._vk_descriptor_pool,
                descriptorSetCount=self.n_images,
                pSetLayouts=[self._vk_descriptor_set_layout,] * self.n_images
            )
        )
        if update:
            self.update()

    def update(self):
        for i_image, _vk_descriptor_set in enumerate(self._vk_descriptor_sets): 
            _vk_writes = []
            for i_binding, obj in enumerate(self.objects):
                _vk_writes.append(obj.get_write_descriptor(_vk_descriptor_set, i_image, i_binding))  # is there anything else get_write_descriptor would need?
            vk.vkUpdateDescriptorSets(self.app._vk_device, len(_vk_writes), _vk_writes, 0, None)

    def destroy(self):
        vk.vkDestroyDescriptorSetLayout(self.app._vk_device, self._vk_descriptor_set_layout, None)

    def __str__(self):
        lines = [f"DescriptorSet ({len(self.objects)} bindings):"]
        for i, obj in enumerate(self.objects):
            lines.append(f"  [{i}]: {obj.__class__.__name__}")
        return "\n".join(lines)
    
    @staticmethod
    def bind(pipeline_layout, command_buffer, descriptor_sets, image_i: int, first_set=0, bind_point=vk.VK_PIPELINE_BIND_POINT_GRAPHICS):
        vk.vkCmdBindDescriptorSets(
            command_buffer._vk_command_buffer,
            bind_point,
            pipeline_layout,
            first_set,
            len(descriptor_sets), [x._vk_descriptor_sets[image_i] for x in descriptor_sets],
            0, None
        )

if __name__ == '__main__':
    # This example is for demonstration and will not run as-is
    # because it requires a valid Vulkan application instance.
    class MockApp: pass
    app = MockApp()
    app._vk_device = None
    app._vk_descriptor_pool = None

    ub = UniformBuffer()
    t1 = Texture()
    t2 = Texture()
    ds = DescriptorSet(app)
    ds.add(ub, stages=vk.VK_SHADER_STAGE_ALL_GRAPHICS)
    ds.add(t1, stages=vk.VK_SHADER_STAGE_FRAGMENT_BIT)
    ds.add(t2, stages=vk.VK_SHADER_STAGE_FRAGMENT_BIT)
    
    print(ds)
    
