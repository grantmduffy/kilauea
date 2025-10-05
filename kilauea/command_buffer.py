import vulkan as vk


class CommandBuffer:

    @staticmethod
    def make_command_buffers(app, n):
        return [CommandBuffer(app, x) for x in vk.vkAllocateCommandBuffers(
            app._vk_device, vk.VkCommandBufferAllocateInfo(
                commandPool=app._vk_command_pool,
                level=getattr(vk, f'VK_COMMAND_BUFFER_LEVEL_PRIMARY'),
                commandBufferCount=n
            )
        )]

    def __init__(self, app, command_buffer):
        self.app = app
        self._vk_command_buffer = command_buffer

    def __enter__(self):
        info = vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
        )
        vk.vkBeginCommandBuffer(self._vk_command_buffer, info)

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        vk.vkEndCommandBuffer(self._vk_command_buffer)
