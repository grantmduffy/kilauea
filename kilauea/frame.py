import vulkan as vk


class Frame:

    def __init__(self, parent):
        self.parent = parent
        self.image_available_semaphore = Semaphore(parent)
        self.render_finished_semaphore = Semaphore(parent)
        self.fence = Fence(parent)

    def recreate_sync_objects(self):
        # Destroy old semaphores
        self.image_available_semaphore.destroy()
        self.render_finished_semaphore.destroy()
        
        # Create new ones
        self.image_available_semaphore = Semaphore(self.parent)
        self.render_finished_semaphore = Semaphore(self.parent)


class Semaphore:
    def __init__(self, app):
        self.app = app
        self._vk_semaphore = vk.vkCreateSemaphore(self.app._vk_device, vk.VkSemaphoreCreateInfo(), None)

    def destroy(self):
        vk.vkDestroySemaphore(self.app._vk_device, self._vk_semaphore, None)


class Fence:

    def __init__(self, app, timeout=1000000000):
        self.app = app
        self.timeout = timeout
        self._vk_fence = vk.vkCreateFence(
            self.app._vk_device, vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT), None
        )
    
    def wait(self):
        vk.vkWaitForFences(
            self.app._vk_device, 
            fenceCount=1, 
            pFences=[self._vk_fence], 
            waitAll=vk.VK_TRUE, 
            timeout=self.timeout
        )

    def reset(self):
        vk.vkResetFences(self.app._vk_device, fenceCount=1, pFences=[self._vk_fence])


class Image:

    def __init__(self, app, render_pass=None, image=None, width=None, height=None, format=None, usage=None):
        self.app = app
        self._vk_image = image
        self.render_pass = render_pass
        self.created_image = image is None
        self.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self.width, self.height = width, height
        self.format = format
        self.usage = usage
        self.app.swapchain.objects.append(self)


    def create(self):
        if self.created_image:
            # Use explicit dimensions if provided, otherwise default to swapchain resolution
            if self.width is None:
                self.width = self.app._vk_extent.width
            if self.height is None:
                self.height = self.app._vk_extent.height
            
            self.format = self.format or self.app.surface_format
            self.usage = self.usage or (vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_SAMPLED_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT)
            
            print(f"Creating user image: {self.width}x{self.height}")

            self._vk_image = vk.vkCreateImage(
                self.app._vk_device,
                vk.VkImageCreateInfo(
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
                    mipLevels=1,
                    arrayLayers=1,
                    format=self.format,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    usage=self.usage,
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                ),
                None
            )
            
            mem_req = vk.vkGetImageMemoryRequirements(self.app._vk_device, self._vk_image)
            mem_type_index = self.app.get_memory_type_index(
                mem_req.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            )
            self._vk_memory = vk.vkAllocateMemory(
                self.app._vk_device,
                vk.VkMemoryAllocateInfo(
                    allocationSize=mem_req.size,
                    memoryTypeIndex=mem_type_index
                ),
                None
            )
            vk.vkBindImageMemory(self.app._vk_device, self._vk_image, self._vk_memory, 0)
        else:
            # For swapchain images, set dimensions from swapchain extent
            self.width = self.app._vk_extent.width
            self.height = self.app._vk_extent.height
            self.format = self.app.surface_format


        self._vk_image_view = vk.vkCreateImageView(
            device=self.app._vk_device,
            pCreateInfo=vk.VkImageViewCreateInfo(
                image=self._vk_image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.format,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1
                )
            ),
            pAllocator=None
        )

        if self.render_pass:
            # Both user images and swapchain images now use scaled resolution
            fb_width = self.width if self.created_image else self.app._vk_extent.width
            fb_height = self.height if self.created_image else self.app._vk_extent.height
            
            self._vk_framebuffer = vk.vkCreateFramebuffer(
                self.app._vk_device, 
                vk.VkFramebufferCreateInfo(
                    renderPass=self.render_pass._vk_render_pass,
                    attachmentCount=1,
                    pAttachments=[self._vk_image_view,],
                    width=fb_width,
                    height=fb_height,
                    layers=1
                ), None
            )
        else:
            self._vk_framebuffer = None

    def destroy(self):
        if hasattr(self, '_vk_framebuffer') and self._vk_framebuffer:
            vk.vkDestroyFramebuffer(self.app._vk_device, self._vk_framebuffer, None)
            self._vk_framebuffer = None
        if hasattr(self, '_vk_image_view') and self._vk_image_view:
            vk.vkDestroyImageView(self.app._vk_device, self._vk_image_view, None)
            self._vk_image_view = None
        if self.created_image:
            if hasattr(self, '_vk_image') and self._vk_image:
                vk.vkDestroyImage(self.app._vk_device, self._vk_image, None)
                self._vk_image = None
            if hasattr(self, '_vk_memory') and self._vk_memory:
                vk.vkFreeMemory(self.app._vk_device, self._vk_memory, None)
                self._vk_memory = None

    def transition_layout(self, command_buffer, new_layout):
        if new_layout == self.layout:
            return

        barrier = vk.VkImageMemoryBarrier(
            oldLayout=self.layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self._vk_image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            )
        )

        # Define source and destination stages based on layout transition
        if self.layout == vk.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            source_stage = vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        elif self.layout == vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL and new_layout == vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            barrier.dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT
            source_stage = vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        elif self.layout == vk.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == vk.VK_IMAGE_LAYOUT_GENERAL:
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT
            source_stage = vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        elif self.layout == vk.VK_IMAGE_LAYOUT_GENERAL and new_layout == vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            barrier.srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT
            barrier.dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT
            source_stage = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
        else:
            # Generic transition - can be expanded
            barrier.srcAccessMask = 0
            barrier.dstAccessMask = 0
            source_stage = vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
            destination_stage = vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT

        vk.vkCmdPipelineBarrier(
            command_buffer._vk_command_buffer,
            source_stage, destination_stage,
            0, 0, None, 0, None, 1, [barrier,]
        )

        self.layout = new_layout


class Texture:

    def __init__(self, app, image, storage=False):
        self.app = app
        self.image = image
        self.storage = storage
        self.app.swapchain.objects.append(self)

    def create(self):
        if not self.storage:
            self._vk_sampler = vk.vkCreateSampler(
                self.app._vk_device,
                vk.VkSamplerCreateInfo(
                    magFilter=vk.VK_FILTER_LINEAR,
                    minFilter=vk.VK_FILTER_LINEAR,
                    addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
                    anisotropyEnable=vk.VK_FALSE,
                    borderColor=vk.VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                    unnormalizedCoordinates=vk.VK_FALSE,
                    compareEnable=vk.VK_FALSE,
                    compareOp=vk.VK_COMPARE_OP_ALWAYS,
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
                ),
                None
            )
        else:
            self._vk_sampler = None

    def destroy(self):
        if hasattr(self, '_vk_sampler') and self._vk_sampler:
            vk.vkDestroySampler(self.app._vk_device, self._vk_sampler, None)
            self._vk_sampler = None
