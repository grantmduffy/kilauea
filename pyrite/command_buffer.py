import vulkan as vk


class CommandBuffer:

    @staticmethod
    def make_command_buffers(app, n):
        return [CommandBuffer(x) for x in vk.vkAllocateCommandBuffers(
            app._vk_device, vk.VkCommandBufferAllocateInfo(
                commandPool=app._vk_command_pool,
                level=getattr(vk, f'VK_COMMAND_BUFFER_LEVEL_PRIMARY'),
                commandBufferCount=n
            )
        )]

    def __init__(self, command_buffer):
        self._vk_command_buffer = command_buffer

    def __enter__(self):
        info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
        )
        vk.vkBeginCommandBuffer(self._vk_command_buffer, info)

    def __exit__(self, exc_type, exc_value, traceback):
        vk.vkEndCommandBuffer(self._vk_command_buffer)
