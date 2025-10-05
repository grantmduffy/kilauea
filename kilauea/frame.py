import vulkan as vk


class Frame:

    def __init__(self, parent):
        self.parent = parent
        self.image_available_semaphore = Semaphore(parent)
        self.pass_semaphores = {}  # keys: (semaphore_signaler, semaphore_consumer)
        self.fence = Fence(parent)
        self.image_available_semaphore = Semaphore(self.parent)

    # def create(self):
    #     self.image_available_semaphore = Semaphore(self.parent)
    #     for k in self.pass_semaphores:
    #         self.pass_semaphores[k] = Semaphore(self.parent)

    def get_semaphores_signaled_by(self, o):
        return [v for k, v in self.pass_semaphores.items() if k[0] == o]
    
    def get_semaphores_consumed_by(self, o):
        return [v for k, v in self.pass_semaphores.items() if k[1] == o]


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

    def __init__(
                self, app, render_pass=None, 
                images=None, n_images=None, 
                extent=None, width=None, height=None, 
                format=None, usage=None
            ):
        self.app = app
        self._vk_images = images
        self.n_images = n_images or len(self._vk_images)
        self.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self.extent = extent or Extent(width, height)
        self.format = format
        self.usage = usage
        # self.app.swapchain.objects.append(self)  # might not need this
        self.create()

    def create(self):
        if self._vk_images is None:
            
            self.format = self.format or self.app.surface_format
            self.usage = self.usage or (vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_SAMPLED_BIT | vk.VK_IMAGE_USAGE_STORAGE_BIT)
            
            self._vk_images = [vk.vkCreateImage(
                self.app._vk_device,
                vk.VkImageCreateInfo(
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    extent=vk.VkExtent3D(width=self.extent.width, height=self.extent.height, depth=1),
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
            ) for _ in range(self.n_images)]
            
            self._vk_memory = []
            for i_image in range(self.n_images):
                mem_req = vk.vkGetImageMemoryRequirements(self.app._vk_device, self._vk_images[i_image])
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
                vk.vkBindImageMemory(self.app._vk_device, self._vk_images[i_image], self._vk_memory, 0)
        else:
            # For swapchain images, set dimensions from swapchain extent
            self.format = self.app.surface_format

        self._vk_image_views = [vk.vkCreateImageView(
            device=self.app._vk_device,
            pCreateInfo=vk.VkImageViewCreateInfo(
                image=img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.format,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1
                )
            ),
            pAllocator=None
        ) for img in self._vk_images]

    def destroy(self):
        # TODO: Fix this later
        pass

    def transition_layout(self, command_buffer, new_layout, image_i=0):
        if new_layout == self.layout:
            return

        barrier = vk.VkImageMemoryBarrier(
            oldLayout=self.layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=self._vk_images[image_i],
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            )
        )

        # TODO: check this logic
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
        self.create()

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
        # if hasattr(self, '_vk_sampler') and self._vk_sampler:
        #     vk.vkDestroySampler(self.app._vk_device, self._vk_sampler, None)
        #     self._vk_sampler = None
        pass

    def get_layout_binding(self, binding: int, stages):
        descriptor_type = (
            vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE if self.storage
            else vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        )
        return vk.VkDescriptorSetLayoutBinding(
            binding=binding,
            descriptorType=descriptor_type,
            descriptorCount=1,
            stageFlags=stages,
        )

    def get_write_descriptor(self, _vk_descriptor_set, i_image: int, i_binding: int):
        image_layout = (
            vk.VK_IMAGE_LAYOUT_GENERAL if self.storage
            else vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        )
        sampler = None if self.storage else self._vk_sampler

        image_info = vk.VkDescriptorImageInfo(
            sampler=sampler,
            imageView=self.image._vk_image_views[i_image],
            imageLayout=image_layout
        )
        descriptor_type = (
            vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE if self.storage
            else vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        )

        write_set = vk.VkWriteDescriptorSet(
            dstSet=_vk_descriptor_set,
            dstBinding=i_binding,
            dstArrayElement=0,
            descriptorType=descriptor_type,
            descriptorCount=1,
            pImageInfo=image_info
        )
        return write_set


class Extent:

    def __init__(self, width=None, height=None, vk_extent=None):
        self._vk_extent = vk_extent or vk.VkExtent2D(width=width, height=height)

    def __getattr__(self, attr):
        if attr == 'width':
            return self._vk_extent.width
        if attr == 'height':
            return self._vk_extent.height
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
