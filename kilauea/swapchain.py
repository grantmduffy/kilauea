import vulkan as vk
from .command_buffer import CommandBuffer
from .frame import Image
from .draw import Pass
from .frame import Semaphore, Extent


class Swapchain:

    def __init__(self, app, n_images, composite_alpha='opaque', wait_for=None, extent=None):
        self.n_images = n_images
        self.current_image_i = -1
        self._vk_swapchain = None
        self.app = app
        self.objects = []
        self.composite_alpha = composite_alpha
        self.wait_for = wait_for or []
        self.extent = extent or Extent()

        
    def get_next_image(self, frame, timeout=1000000000):
        self.current_image_i = vk.vkGetDeviceProcAddr(self.app._vk_device, 'vkAcquireNextImageKHR')(
            self.app._vk_device, swapchain=self._vk_swapchain, timeout=timeout,
            semaphore=frame.image_available_semaphore._vk_semaphore, fence=vk.VK_NULL_HANDLE
        )
        return self.current_image_i

    def present_image(self, image_i, presentation_semaphores):
        vk.vkGetDeviceProcAddr(self.app._vk_device, 'vkQueuePresentKHR')(
            self.app._vk_queue, vk.VkPresentInfoKHR(
                waitSemaphoreCount=len(presentation_semaphores), pWaitSemaphores=[x._vk_semaphore for x in presentation_semaphores],
                swapchainCount=1, pSwapchains=[self._vk_swapchain],
                pImageIndices=[image_i]
            )
        )

    def present_wait_for(self, passes: list[Pass]):
        self.wait_for = passes
        for obj in self.wait_for:
            for frame in self.app.frames:
                frame.pass_semaphores[(obj, self)] = Semaphore(self.app)

    def create(self, width, height):

        for x in self.wait_for:
            for i in range(self.n_frames):
                self.app.frames[i].pass_semaphores[(x, self)] = Semaphore(self.app)

        supported_surface_capabilities = vk.vkGetInstanceProcAddr(
            self.app._vk_instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR'
        )(self.app._vk_physical_device, self.app._vk_surface, None)

        # Update swapchain extent to match new window size
        if supported_surface_capabilities.currentExtent.width != 0xFFFFFFFF:
            new_extent = supported_surface_capabilities.currentExtent
        else:
            new_extent = vk.VkExtent2D(
                width=max(min(width, supported_surface_capabilities.maxImageExtent.width), supported_surface_capabilities.minImageExtent.width), 
                height=max(min(height, supported_surface_capabilities.maxImageExtent.height), supported_surface_capabilities.minImageExtent.height)
            )
        self.extent._vk_extent = new_extent

        # First create the Vulkan swapchain
        supported_present_modes = vk.vkGetInstanceProcAddr(
            self.app._vk_instance, 'vkGetPhysicalDeviceSurfacePresentModesKHR'
        )(self.app._vk_physical_device, self.app._vk_surface)
        if vk.VK_PRESENT_MODE_IMMEDIATE_KHR in supported_present_modes:
            self.present_mode = vk.VK_PRESENT_MODE_IMMEDIATE_KHR
        elif vk.VK_PRESENT_MODE_MAILBOX_KHR in supported_present_modes:
            self.present_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR
        else:
            self.present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
        if self.app.graphics_queue_family_i == self.app.present_queue_family_i:
            image_sharing_mode = vk.VK_SHARING_MODE_EXCLUSIVE
            queue_family_index_count = 0
            queue_family_indices = None
        else:
            image_sharing_mode = vk.VK_SHARING_MODE_CONCURRENT
            queue_family_index_count = 2
            queue_family_indices = [self.app.graphics_queue_family_i, self.app.present_queue_family_i]
        # Use the window extent for swapchain image creation
        print(f"Creating swapchain with extent: {self.extent.width}x{self.extent.height}")
        
        self._vk_swapchain = vk.vkGetDeviceProcAddr(self.app._vk_device, 'vkCreateSwapchainKHR')(self.app._vk_device, vk.VkSwapchainCreateInfoKHR(
            surface=self.app._vk_surface,
            minImageCount=self.n_images,
            imageFormat=self.app.surface_format,
            imageColorSpace=self.app.color_space,
            imageExtent=self.app._vk_extent,  # Use window extent for swapchain
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 
            imageSharingMode=image_sharing_mode,
            queueFamilyIndexCount=queue_family_index_count,
            pQueueFamilyIndices=queue_family_indices,
            preTransform=self.app.supported_surface_capabilities.currentTransform,
            compositeAlpha=getattr(vk, f'VK_COMPOSITE_ALPHA_{self.composite_alpha.upper()}_BIT_KHR'),
            presentMode=self.present_mode,
            clipped=vk.VK_TRUE
        ), None)

        # Create swapchain images and internal objects BEFORE user objects
        self.swapchain_render_pass = Pass(self.app)
        self.objects.pop()  # remove the pass so it's not recreated with other objects
        self.swapchain_render_pass.create()
        self.images = Image(
            self.app, 
            render_pass=self.swapchain_render_pass, 
            images=vk.vkGetDeviceProcAddr(
                self.app._vk_device, 'vkGetSwapchainImagesKHR'
            )(self.app._vk_device, self._vk_swapchain)
        )

    def destroy(self):
        pass
