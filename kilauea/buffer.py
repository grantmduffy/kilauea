import vulkan as vk
import numpy as np


class Buffer:

    def __init__(self, app, data: np.ndarray=None, usage='vertex'):
        self.app = app
        self.data = data
        self.usage = getattr(vk, f'VK_BUFFER_USAGE_{usage.upper()}_BUFFER_BIT')
        self._vk_buffer = vk.vkCreateBuffer(
            self.app._vk_device,
            vk.VkBufferCreateInfo(
                size=self.data.nbytes,
                usage=self.usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            ),
            None
        )
        self.memory_req = vk.vkGetBufferMemoryRequirements(self.app._vk_device, self._vk_buffer)
        self.memory_type_index = self.app.get_memory_type_index(
            self.memory_req.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        self._vk_memory = vk.vkAllocateMemory(
            self.app._vk_device,
            vk.VkMemoryAllocateInfo(
                allocationSize=self.memory_req.size,
                memoryTypeIndex=self.memory_type_index
            ),
            None
        )
        vk.vkBindBufferMemory(self.app._vk_device, self._vk_buffer, self._vk_memory, 0)
        self.write(self.data)

    def write(self, data):
        mem_ptr = vk.vkMapMemory(self.app._vk_device, self._vk_memory, 0, self.memory_req.size, 0)
        vk.ffi.memmove(mem_ptr, data.tobytes(), data.nbytes)
        vk.vkUnmapMemory(self.app._vk_device, self._vk_memory)
