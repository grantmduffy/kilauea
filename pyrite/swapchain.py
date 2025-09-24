import vulkan as vk
from .command_buffer import CommandBuffer
from .frame import Image
from .draw import Pass


class Swapchain:

    def __init__(self, parent, n_images, composite_alpha='opaque'):
        self.n_images = n_images
        self.current_image = -1
        self._vk_swapchain = None
        self.parent = parent
        self.objects = []
        self.composite_alpha = composite_alpha

        
    def get_next_image(self, frame, timeout=1000000000):
        self.current_image = vk.vkGetDeviceProcAddr(self.parent._vk_device, 'vkAcquireNextImageKHR')(
            self.parent._vk_device, swapchain=self._vk_swapchain, timeout=timeout,
            semaphore=frame.image_available_semaphore._vk_semaphore, fence=vk.VK_NULL_HANDLE
        )
        return self.current_image, self.command_buffers[self.current_image]

    def present_image(self, image_i, signal_semaphore):
        vk.vkGetDeviceProcAddr(self.parent._vk_device, 'vkQueuePresentKHR')(
            self.parent._vk_queue, vk.VkPresentInfoKHR(
                waitSemaphoreCount=1, pWaitSemaphores=[signal_semaphore._vk_semaphore],
                swapchainCount=1, pSwapchains=[self._vk_swapchain],
                pImageIndices=[image_i]
            )
        )

    def create(self):
        # First create the Vulkan swapchain
        supported_present_modes = vk.vkGetInstanceProcAddr(
            self.parent._vk_instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR'
        )(self.parent._vk_physical_device, self.parent._vk_surface)
        if vk.VK_PRESENT_MODE_IMMEDIATE_KHR in supported_present_modes:
            self.present_mode = vk.VK_PRESENT_MODE_IMMEDIATE_KHR
        elif vk.VK_PRESENT_MODE_MAILBOX_KHR in supported_present_modes:
            self.present_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR
        else:
            self.present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if self.parent.graphics_queue_family_i == self.parent.present_queue_family_i:
            image_sharing_mode = vk.VK_SHARING_MODE_EXCLUSIVE
            queue_family_index_count = 0
            queue_family_indices = None
        else:
            image_sharing_mode = vk.VK_SHARING_MODE_CONCURRENT
            queue_family_index_count = 2
            queue_family_indices = [self.parent.graphics_queue_family_i, self.parent.present_queue_family_i]
        self._vk_swapchain = vk.vkGetDeviceProcAddr(self.parent._vk_device, 'vkCreateSwapchainKHR')(self.parent._vk_device, vk.VkSwapchainCreateInfoKHR(
            surface=self.parent._vk_surface,
            minImageCount=self.n_images,
            imageFormat=self.parent.surface_format,
            imageColorSpace=self.parent.color_space,
            imageExtent=self.parent._vk_extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
            imageSharingMode=image_sharing_mode,
            queueFamilyIndexCount=queue_family_index_count,
            pQueueFamilyIndices=queue_family_indices,
            preTransform=self.parent.supported_surface_capabilities.currentTransform,
            compositeAlpha=getattr(vk, f'VK_COMPOSITE_ALPHA_{self.composite_alpha.upper()}_BIT_KHR'),
            presentMode=self.present_mode,
            clipped=vk.VK_TRUE
        ), None)

        # Create swapchain images and internal objects BEFORE user objects
        self.command_buffers = CommandBuffer.make_command_buffers(self.parent, self.n_images)
        self.swapchain_render_pass = Pass(self.parent)
        self.objects.pop()  # remove the pass so it's not recreated with other objects
        self.swapchain_render_pass.create()
        self.images = [
            Image(self.parent, render_pass=self.swapchain_render_pass, image=x) 
            for x in vk.vkGetDeviceProcAddr(
                self.parent._vk_device, 'vkGetSwapchainImagesKHR'
            )(self.parent._vk_device, self._vk_swapchain)
        ]
        
        # Remove swapchain images from objects list so they're not recreated with user objects
        # (they were added by Image.__init__)
        for _ in range(len(self.images)):
            self.objects.pop()
        
        # Create the swapchain images (image views and framebuffers)
        for image in self.images:
            image.create()

        # NOW create user objects - swapchain images are available for reference
        for o in self.objects:
            o.create()

    def destroy(self):
        for o in self.objects:
            o.destroy()
        for image in self.images:
            image.destroy()
        if hasattr(self, 'swapchain_render_pass'):
            self.swapchain_render_pass.destroy()
        if self._vk_swapchain:
            vk.vkGetDeviceProcAddr(self.parent._vk_device, 'vkDestroySwapchainKHR')(self.parent._vk_device, self._vk_swapchain, None)
            self._vk_swapchain = None

    def get_images(self):
        return zip(self.images, self.command_buffers)
