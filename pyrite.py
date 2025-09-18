"""
pyrite is a graphics engine built on vulkan and python

goals of pyrite:
1. Take vulkan patterns and map them to easy-to-use python patterns. Take advantage of python features
   like __enter__()/__exit__(), and getattr() to make it powerful but simple to implement and understand
2. Manage swapchain/windowing so that you can implement a full vulkan app quickly, ex. MyApp
3. Don't add too much fluff (for now), don't worry about corner cases or excessive asserts
4. Work in progress, as I implement, move things around until they make sense
5. True modularity. Group things and put them in heirarchys so it's obvious to the user
"""


import vulkan as vk
import glfw
from threading import Thread
import numpy as np
from pathlib import Path
import subprocess
import tempfile


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


class Swapchain:

    def __init__(self, parent, n_images, composite_alpha='opaque'):
        self.n_images = n_images
        self.current_image = -1
        self._vk_swapchain = None
        self.parent = parent

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
            compositeAlpha=getattr(vk, f'VK_COMPOSITE_ALPHA_{composite_alpha.upper()}_BIT_KHR'),
            presentMode=self.present_mode,
            clipped=vk.VK_TRUE
        ), None)

        # create images
        self.command_buffers = CommandBuffer.make_command_buffers(self.parent, self.n_images)
        self.swapchain_render_pass = Pass(self.parent)
        self.images = [
            Image(self.parent, x, self.swapchain_render_pass) 
            for x in vk.vkGetDeviceProcAddr(
                self.parent._vk_device, 'vkGetSwapchainImagesKHR'
            )(self.parent._vk_device, self._vk_swapchain)
        ]
        
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

    def get_images(self):
        return zip(self.images, self.command_buffers)


class Semaphore:
    def __init__(self, app):
        self._vk_semaphore = vk.vkCreateSemaphore(app._vk_device, vk.VkSemaphoreCreateInfo(), None)


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


class Shader:

    def __init__(self, app, source: str | Path, stage_name: str=None):
        self.app = app
        self.source = source
        self.stage_name = stage_name
        self.code = Path(source).read_text()
        self.spirv = self.compile()
        self._vk_module = vk.vkCreateShaderModule(
            self.app._vk_device,
            vk.VkShaderModuleCreateInfo(
                codeSize=len(self.spirv) * 4,
                pCode=self.spirv.tobytes()
            ),
            None
        )
        self._vk_stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=getattr(vk, f'VK_SHADER_STAGE_{self.stage_name.upper()}_BIT'),
            module=self._vk_module, pName='main'
        )

    def compile(self):
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.glsl') as infile, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.spv') as outfile:
            
            infile.write(self.code)
            infile.flush()

            try:
                result = subprocess.run(
                    [
                        'glslc',
                        f'-fshader-stage={self.stage_name}',
                        infile.name,
                        '-o',
                        outfile.name
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Shader {self.stage_name} compiled successfully")
                if result.stderr:
                    print(f"Shader warnings: {result.stderr}")
            except FileNotFoundError:
                raise RuntimeError("glslc not found. Please install the Vulkan SDK and ensure glslc is in your PATH.")
            except subprocess.CalledProcessError as e:
                print(f"Shader compilation failed for {self.stage_name}:")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                raise RuntimeError(f"Shader compilation failed:\n{e.stderr}")

            outfile.seek(0)
            spirv = np.fromfile(outfile, dtype=np.uint32)
            print(f"SPIRV size for {self.stage_name}: {len(spirv)} words")

        return spirv


class Drawable:

    VERTEX_FORMAT_MAP = {
        'vec2': (vk.VK_FORMAT_R32G32_SFLOAT, 8),
        'vec3': (vk.VK_FORMAT_R32G32B32_SFLOAT, 12),
        'vec4': (vk.VK_FORMAT_R32G32B32A32_SFLOAT, 16),
    }
    
    def __init__(self, app, vertices: np.ndarray, vertex_shader: Shader | str | Path, fragment_shader: Shader | str, 
                 render_pass, indices: np.ndarray=None, targets=None, vertex_attributes=None, uniforms=None):
        self.app = app
        self.vertices = Buffer(app, vertices)
        self.indices = Buffer(indices) if indices is not None else None
        self.vertex_shader = Shader(app, vertex_shader, 'vertex') if not isinstance(vertex_shader, Shader) else vertex_shader
        self.fragment_shader = Shader(app, fragment_shader, 'fragment') if not isinstance(fragment_shader, Shader) else fragment_shader
        self.targets = targets
        self.vertex_attributes = vertex_attributes or []
        self.render_pass = render_pass
        self.uniforms = uniforms

        # infer vertex binding description from attributes list
        stride = sum(self.VERTEX_FORMAT_MAP[attr][1] for attr in self.vertex_attributes)
        vertex_binding_description = vk.VkVertexInputBindingDescription(
            binding=0, stride=stride, inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX
        )
        
        attribute_descriptions = []
        offset = 0
        for i, attr in enumerate(self.vertex_attributes):
            fmt, size = self.VERTEX_FORMAT_MAP[attr]
            attribute_descriptions.append(vk.VkVertexInputAttributeDescription(
                binding=0, location=i, format=fmt, offset=offset
            ))
            offset += size

        vs_inputs = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertexBindingDescriptionCount=1, pVertexBindingDescriptions=[vertex_binding_description],
            vertexAttributeDescriptionCount=len(attribute_descriptions), pVertexAttributeDescriptions=attribute_descriptions
        )

        # TODO: make this dynamic to different topologies
        input_assembly_ci = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE
        )

        viewport_state_ci = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=[self.app._vk_viewport],
            scissorCount=1,
            pScissors=[vk.VkRect2D(
                offset=[0, 0],
                extent=self.app._vk_extent
            )]
        )

        # TODO: make some of this configurable
        rasterizer_ci = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1.0,
            cullMode=vk.VK_CULL_MODE_BACK_BIT,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE
        )

        multisampling_ci = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sampleShadingEnable=vk.VK_FALSE,
            rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT
        )

        # TODO: also make some of this configurable
        color_blend_ci = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logicOpEnable=vk.VK_FALSE,
            attachmentCount=1,
            pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT,
                blendEnable=vk.VK_FALSE
            )],
            blendConstants=[0.0, 0.0, 0.0, 0.0]
        )

        # Create pipeline layout with descriptor set layout if uniforms are provided
        descriptor_set_layouts = []
        if self.uniforms:
            descriptor_set_layouts.append(self.uniforms.descriptor_set_layout)
        
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.app._vk_device, 
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=len(descriptor_set_layouts),
                pSetLayouts=descriptor_set_layouts if descriptor_set_layouts else None
            ), None
        )

        shader_stages = [self.vertex_shader._vk_stage, self.fragment_shader._vk_stage]
        print(f"Creating pipeline with {len(descriptor_set_layouts)} descriptor set layouts")
        print(f"Pipeline layout: {self.pipeline_layout}")
        
        # Create pipeline - let's check if there are validation issues
        pipeline_create_info = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            stageCount=len(shader_stages),
            pStages=shader_stages,
            pVertexInputState=vs_inputs,
            pInputAssemblyState=input_assembly_ci,
            pViewportState=viewport_state_ci,
            pRasterizationState=rasterizer_ci,
            pMultisampleState=multisampling_ci,
            pDepthStencilState=None,
            pColorBlendState=color_blend_ci,
            layout=self.pipeline_layout,
            renderPass=self.render_pass._vk_render_pass,
            subpass=0
        )
        
        print("About to create graphics pipeline...")
        print(f"Shader stages: {len(shader_stages)}")
        print(f"Vertex shader module: {self.vertex_shader._vk_module}")
        print(f"Fragment shader module: {self.fragment_shader._vk_module}")
        print(f"Render pass: {self.render_pass._vk_render_pass}")
        
        try:
            print("Creating pipeline directly...")
            pipeline_result = vk.vkCreateGraphicsPipelines(
                self.app._vk_device, vk.VK_NULL_HANDLE, 1, 
                pipeline_create_info, None
            )
            
            print(f"Pipeline result: {pipeline_result}")
            print(f"Pipeline result type: {type(pipeline_result)}")
            
            # Extract pipeline from result
            if hasattr(pipeline_result, '__len__') and len(pipeline_result) > 0:
                self.pipeline = pipeline_result[0]
            else:
                self.pipeline = pipeline_result
                
            print(f"Pipeline extracted: {self.pipeline}")
            
            # Simple validation - check if it's NULL
            if str(self.pipeline).find('NULL') != -1:
                print("ERROR: Pipeline creation returned NULL")
                self.pipeline = None
            else:
                print(f"Pipeline created successfully: {self.pipeline}")
                
        except Exception as e:
            print(f"Pipeline creation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            self.pipeline = None
    
    def draw(self, command_buffer, frame_index=None):
            
        vk.vkCmdBindPipeline(command_buffer._vk_command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline)
        
        # Bind descriptor sets if uniforms are available
        if self.uniforms and frame_index is not None:
            vk.vkCmdBindDescriptorSets(
                command_buffer._vk_command_buffer,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipeline_layout,
                0,  # firstSet
                1,  # descriptorSetCount
                [self.uniforms.descriptor_sets[frame_index]],
                0,  # dynamicOffsetCount
                None  # pDynamicOffsets
            )
        
        vk.vkCmdBindVertexBuffers(command_buffer._vk_command_buffer, 0, 1, [self.vertices._vk_buffer], [0])
        vk.vkCmdDraw(command_buffer._vk_command_buffer, len(self.vertices.data), 1, 0, 0)


class Pass:
    
    def __init__(self, app, clear_color=(0, 0, 0, 0)):
        self.app = app
        self.clear_color = clear_color

        # TODO: dynamically create color attachment
        color_attachment = vk.VkAttachmentDescription(
            format=self.app.surface_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        )
        color_attachment_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=color_attachment_ref
        )
        self._vk_render_pass = vk.vkCreateRenderPass(
            self.app._vk_device, 
            vk.VkRenderPassCreateInfo(
                attachmentCount=1,
                pAttachments=color_attachment,
                subpassCount=1,
                pSubpasses=subpass
            ), None
        )
        self._vk_clear_value = vk.VkClearValue(vk.VkClearColorValue(self.clear_color))

    def start(self, command_buffer, target_image):
        return PassContext(self, command_buffer, target_image)


class PassContext:

    def __init__(self, render_pass, command_buffer, target_image):
        self.render_pass = render_pass
        self.command_buffer = command_buffer
        self.target_image = target_image

    def __enter__(self):
        vk.vkCmdBeginRenderPass(
            self.command_buffer._vk_command_buffer, vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass._vk_render_pass, framebuffer=self.target_image._vk_framebuffer, 
                renderArea=vk.VkRect2D(offset=[0, 0], extent=self.render_pass.app._vk_extent), 
                clearValueCount=1, pClearValues=self.render_pass._vk_clear_value
            ),
            getattr(vk, f'VK_SUBPASS_CONTENTS_INLINE')
        )

    def __exit__(self, *args):
        vk.vkCmdEndRenderPass(self.command_buffer._vk_command_buffer)


class Frame:

    def __init__(self, parent):
        self.parent = parent
        self.image_available_semaphore = Semaphore(parent)
        self.render_finished_semaphore = Semaphore(parent)
        self.fence = Fence(parent)


class Image:

    def __init__(self, app, image, render_pass):
        self.app = app
        self._vk_image = image
        self.render_pass = render_pass

        self._vk_image_view = vk.vkCreateImageView(
            device=self.app._vk_device,
            pCreateInfo=vk.VkImageViewCreateInfo(
                image=self._vk_image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=self.app.surface_format,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1
                )
            ),
            pAllocator=None
        )
        self._vk_framebuffer = vk.vkCreateFramebuffer(
            self.app._vk_device, 
            vk.VkFramebufferCreateInfo(
                renderPass=self.render_pass._vk_render_pass,
                attachmentCount=1,
                pAttachments=[self._vk_image_view,],
                width=self.app._vk_extent.width,
                height=self.app._vk_extent.height,
                layers=1
            ), None
        )


class App:

    def __init__(
                self, title='Pyrite', size=(640, 480), n_frames=3, n_images=4, version=(1, 0, 0), 
                engine_name='Pyrite', device_preference=['discrete_gpu', 'integrated_gpu', 'virtual_gpu', 'cpu'],
                surface_format=vk.VK_FORMAT_B8G8R8A8_UNORM, color_space=vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,

            ):
        glfw.init()
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        self.title = title
        self.running = False
        self.graphics_thread = None
        self.current_frame = -1
        self.frame_count = 0
        self.version = vk.VK_MAKE_VERSION(*version)
        self.engine_name = engine_name
        self.device_preference = device_preference
        self.surface_format = surface_format
        self.color_space = color_space
        self.size = size

        self.init_vk()
        self.create_descriptor_pool()  # Create descriptor pool for uniform buffers
        self.swapchain = Swapchain(self, n_images)
        self.frames = tuple(Frame(self) for _ in range(n_frames))
        
        
    def init_vk(self):

        self.window = glfw.create_window(*self.size, self.title, None, None)

        app_info = vk.VkApplicationInfo(
            pApplicationName=self.title,
            applicationVersion=self.version,
            pEngineName=self.engine_name,
            engineVersion=self.version,
            apiVersion=self.version
        )

        extensions = glfw.get_required_instance_extensions()
        extensions.append(vk.VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        supported_extensions = [e.extensionName for e in vk.vkEnumerateInstanceExtensionProperties(None)]
        for e in extensions:
            if e not in supported_extensions:
                raise Exception(f'Extension {e} is not supported')
        
        # setup and check layers
        enabled_layers = [
            'VK_LAYER_KHRONOS_validation',
        ]
        supported_layers = [l.layerName for l in vk.vkEnumerateInstanceLayerProperties()]
        for l in enabled_layers:
            if l not in supported_layers:
                raise Exception(f'Layer {l} is not supported')

        self._vk_instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(
                pApplicationInfo=app_info,
                enabledLayerCount=len(enabled_layers),
                ppEnabledLayerNames=enabled_layers,
                enabledExtensionCount=len(extensions),
                ppEnabledExtensionNames=extensions
            ), None
        )

        surface = vk.ffi.new('VkSurfaceKHR *')
        glfw.create_window_surface(self._vk_instance, self.window, None, surface)
        self._vk_surface = surface[0]

        creation_function = vk.vkGetInstanceProcAddr(self._vk_instance, 'vkCreateDebugReportCallbackEXT')
        self._vk_debug_messenger = creation_function(self._vk_instance, vk.VkDebugReportCallbackCreateInfoEXT(
            flags = vk.VK_DEBUG_REPORT_ERROR_BIT_EXT | vk.VK_DEBUG_REPORT_WARNING_BIT_EXT,
            pfnCallback=self.debug_callback
        ), None)

        available_devices = vk.vkEnumeratePhysicalDevices(self._vk_instance)
        possible_types = ['cpu', 'discrete_gpu', 'integrated_gpu', 'virtual_gpu']
        possible_types = {x: getattr(vk, f'VK_PHYSICAL_DEVICE_TYPE_{x.upper()}') for x in possible_types}
        self._vk_physical_device = None
        for device_type in self.device_preference:
            for d in available_devices:
                if vk.vkGetPhysicalDeviceProperties(d).deviceType == possible_types[device_type]:
                    self._vk_physical_device = d
                    break
            if self._vk_physical_device is not None:
                break
        if self._vk_physical_device is None:
            raise Exception('No suitable device found')
        
        w, h = glfw.get_framebuffer_size(self.window)
        self.supported_surface_capabilities = vk.vkGetInstanceProcAddr(
            self._vk_instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR'
        )(self._vk_physical_device, self._vk_surface, None)
        self._vk_extent = vk.VkExtent2D(
            width=max(min(w, self.supported_surface_capabilities.maxImageExtent.width), self.supported_surface_capabilities.minImageExtent.width), 
            height=max(min(h, self.supported_surface_capabilities.maxImageExtent.height), self.supported_surface_capabilities.minImageExtent.height)
        )

        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(self._vk_physical_device)
        self.graphics_queue_family_i = None
        self.present_queue_family_i = None
        for i, q in enumerate(queue_families):
            if self.graphics_queue_family_i is None and q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                self.graphics_queue_family_i = i
            if self.present_queue_family_i is None and vk.vkGetInstanceProcAddr(self._vk_instance, 'vkGetPhysicalDeviceSurfaceSupportKHR')(self._vk_physical_device, i, self._vk_surface):
                self.present_queue_family_i = i
        unique_queue_i = list({
            self.graphics_queue_family_i, 
            self.present_queue_family_i, 
        })
        queue_create_info = [vk.VkDeviceQueueCreateInfo(
            queueFamilyIndex=fam,
            queueCount=1,
            pQueuePriorities=[1.0]
        ) for fam in unique_queue_i]
        device_extensions = [
            vk.VK_KHR_SWAPCHAIN_EXTENSION_NAME
        ]

        self._vk_device = vk.vkCreateDevice(
            self._vk_physical_device, 
            [vk.VkDeviceCreateInfo(
                queueCreateInfoCount = len(queue_create_info),
                pQueueCreateInfos = queue_create_info,
                enabledExtensionCount = len(device_extensions),
                ppEnabledExtensionNames=device_extensions,
                pEnabledFeatures = [vk.VkPhysicalDeviceFeatures(),],
                enabledLayerCount = len(enabled_layers),
                ppEnabledLayerNames = enabled_layers
            ),], 
            None
        )

        self._vk_queue = vk.vkGetDeviceQueue(self._vk_device, self.graphics_queue_family_i, 0)
        self._vk_command_pool = vk.vkCreateCommandPool(
            self._vk_device, vk.VkCommandPoolCreateInfo(
                queueFamilyIndex=self.graphics_queue_family_i,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
            ), None
        )

        self._vk_viewport = vk.VkViewport(
            x=0, y=0, width=self._vk_extent.width, height=self._vk_extent.height,
            minDepth=0.0, maxDepth=1.0
        )

    def get_memory_type_index(self, type_bits, properties):
        mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self._vk_physical_device)
        for i in range(mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) and ((mem_props.memoryTypes[i].propertyFlags & properties) == properties):
                return i
        return None

    def create_descriptor_pool(self, max_sets=100):
        """Create a descriptor pool for uniform buffer descriptors"""
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=max_sets
            )
        ]
        
        self._vk_descriptor_pool = vk.vkCreateDescriptorPool(
            self._vk_device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                maxSets=max_sets,
                poolSizeCount=len(pool_sizes),
                pPoolSizes=pool_sizes
            ),
            None
        )
        return self._vk_descriptor_pool

    @staticmethod
    def debug_callback(*args):
        print("Validation:", args[5])
        print("Details:", args[6])
        # Don't raise exception to see more details
        return 0

    def get_next_frame(self):
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        frame = self.frames[self.current_frame]
        frame.fence.wait()
        frame.fence.reset()
        return frame

    def main_loop(self):
        pass

    def record_draw_commands(self):
        for image, command_buffer in self.swapchain.get_images():
            with command_buffer:
                self.draw(command_buffer, image)

    def draw(self, command_buffer, image):
        raise NotImplementedError()

    def graphics_loop(self):
        while self.running:
            frame = self.get_next_frame()
            image, command_buffer = self.swapchain.get_next_image(frame)
            self.submit_commands(command_buffer, frame)
            self.swapchain.present_image(image, frame.render_finished_semaphore)
            self.frame_count += 1
    
    def submit_commands(self, command_buffer, frame):
        vk.vkQueueSubmit(self._vk_queue, 1, vk.VkSubmitInfo(
                waitSemaphoreCount=1, pWaitSemaphores=[frame.image_available_semaphore._vk_semaphore],
                pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,],
                commandBufferCount=1, pCommandBuffers=[command_buffer._vk_command_buffer],
                signalSemaphoreCount=1, pSignalSemaphores=[frame.render_finished_semaphore._vk_semaphore]
            ), fence=frame.fence._vk_fence)

    def run(self):

        # record draw calls
        self.record_draw_commands()

        # start running, including starting graphics thread
        self.running = True
        self.graphics_thread = Thread(target=self.graphics_loop)
        self.graphics_thread.start()

        # run the main loop
        while self.running:
            self.running = not glfw.window_should_close(self.window)
            glfw.poll_events()
            self.main_loop()
        
        # shutdown
        self.graphics_thread.join()
        self.destroy()
    
    def destroy(self):
        glfw.terminate()


class Uniform:

    # Vulkan alignment requirements for different data types
    ALIGNMENT_MAP = {
        np.float32: 4,
        np.int32: 4,
        np.uint32: 4,
    }
    
    # Size requirements for common GLSL types
    TYPE_SIZE_MAP = {
        'float': 4,
        'vec2': 8,
        'vec3': 12,  # but aligns to 16
        'vec4': 16,
        'mat3': 48,  # but aligns to 64 (3 vec4s)
        'mat4': 64,
        'int': 4,
        'ivec2': 8,
        'ivec3': 12,  # but aligns to 16
        'ivec4': 16,
    }

    def __init__(self, data=None, glsl_type=None):
        self.data = data.copy() if data is not None else None  # Store original data
        self.original_data = data.copy() if data is not None else None
        self.glsl_type = glsl_type
        self.np_size = self.data.nbytes if self.data is not None else 0
        self.vk_size = self._calculate_vk_size()
        self.offset = 0  # Will be set by UniformBuffer
        self.mapped_data = None  # Will point to mapped buffer memory
    
    def _calculate_vk_size(self):
        """Calculate the size with proper Vulkan alignment"""
        if self.data is None:
            return 0
            
        # For matrices, ensure proper alignment
        if self.glsl_type == 'mat4':
            return 64  # mat4 is always 64 bytes
        elif self.glsl_type == 'mat3':
            return 64  # mat3 aligns to 64 bytes (treated as 3 vec4s)
        elif self.glsl_type in ['vec3', 'ivec3']:
            return 16  # vec3 aligns to 16 bytes
        elif self.glsl_type == 'float':
            return 4  # float is 4 bytes
        else:
            # For other types, align to next multiple of base alignment
            base_size = self.np_size
            alignment = self._get_alignment()
            return ((base_size + alignment - 1) // alignment) * alignment
    
    def _get_alignment(self):
        """Get alignment requirement for the data type"""
        if self.data is None:
            return 4
        
        # GLSL std140 alignment rules
        if self.glsl_type == 'mat4':
            return 16  # mat4 columns align to 16 bytes
        elif self.glsl_type == 'float':
            return 4   # float aligns to 4 bytes
        elif self.glsl_type in ['vec3', 'ivec3']:
            return 16  # vec3 aligns to 16 bytes
        
        dtype = self.data.dtype.type
        return self.ALIGNMENT_MAP.get(dtype, 4)
    
    def map_to_location(self, mapped_memory, offset):
        """Map this uniform to a location in the persistent buffer"""
        self.offset = offset
        
        # Create numpy array view into the mapped memory
        buffer_view = np.frombuffer(
            mapped_memory, 
            dtype=self.data.dtype, 
            count=self.data.size,
            offset=offset
        ).reshape(self.data.shape)
        
        # Copy original data to mapped location
        buffer_view[:] = self.original_data
        
        # Replace our data with the mapped view
        self.mapped_data = buffer_view
    
    def __getitem__(self, k):
        return self.mapped_data[k] if self.mapped_data is not None else self.data[k]
    
    def __setitem__(self, k, v):
        if self.mapped_data is not None:
            self.mapped_data[k] = v
        else:
            self.data[k] = v


class UniformBuffer:

    def __init__(self, app, data=None):
        self.app = app
        self.uniforms = {}
        self._vk_buffer = None  # Vulkan buffer object
        self._vk_memory = None  # Vulkan memory object
        self.mapped_memory = None  # Persistently mapped memory pointer
        self.total_size = 0
        self.min_uniform_buffer_offset_alignment = None
    
    def __setitem__(self, uniform_name, uniform):
        if self._vk_buffer is not None:
            raise RuntimeError("Cannot add uniforms after buffer has been created")
        self.uniforms[uniform_name] = uniform
    
    def __getitem__(self, uniform_name):
        return self.uniforms[uniform_name]
    
    def add_uniform(self, name, *args, **kwargs):
        if self._vk_buffer is not None:
            raise RuntimeError("Cannot add uniforms after buffer has been created")
        self.uniforms[name] = Uniform(*args, **kwargs)
    
    def _get_uniform_buffer_alignment(self):
        """Get the minimum uniform buffer offset alignment from device properties"""
        if self.min_uniform_buffer_offset_alignment is None:
            device_props = vk.vkGetPhysicalDeviceProperties(self.app._vk_physical_device)
            self.min_uniform_buffer_offset_alignment = device_props.limits.minUniformBufferOffsetAlignment
        return self.min_uniform_buffer_offset_alignment
    
    def _calculate_aligned_offset(self, offset, alignment):
        """Calculate the next aligned offset"""
        return ((offset + alignment - 1) // alignment) * alignment
    
    def create_descriptor_set_layout(self):
        """Create descriptor set layout for this uniform buffer"""
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                # TODO: automatically determine stages
                stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT | vk.VK_SHADER_STAGE_FRAGMENT_BIT, 
                pImmutableSamplers=None
            )
        ]
        
        self.descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
            self.app._vk_device,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings),
                pBindings=bindings
            ),
            None
        )
        return self.descriptor_set_layout
    
    def create_descriptor_set(self, descriptor_pool):
        """Create descriptor sets for each frame in flight"""
        if not hasattr(self, 'descriptor_set_layout'):
            raise RuntimeError("Must create descriptor set layout first")
        
        self.descriptor_sets = vk.vkAllocateDescriptorSets(
            self.app._vk_device,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[self.descriptor_set_layout]
            )
        )
        
        # Update descriptor sets to point to our uniform buffer
        for i, descriptor_set in enumerate(self.descriptor_sets):
            buffer_info = vk.VkDescriptorBufferInfo(
                buffer=self._vk_buffer,
                offset=0,
                range=self.total_size
            )
            
            write_descriptor_set = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=0,
                dstArrayElement=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=1,
                pBufferInfo=[buffer_info]
            )
            
            vk.vkUpdateDescriptorSets(
                self.app._vk_device,
                descriptorWriteCount=1,
                pDescriptorWrites=[write_descriptor_set],
                descriptorCopyCount=0,
                pDescriptorCopies=None
            )
        
        print(f"Created {len(self.descriptor_sets)} descriptor sets for uniform buffer")
        return self.descriptor_sets

    def make_buffer(self):
        """Create the Vulkan buffer and map all uniforms to it"""
        if not self.uniforms:
            raise RuntimeError("No uniforms added to buffer")
        
        if self._vk_buffer is not None:
            raise RuntimeError("Buffer already created")
        
        # Calculate total buffer size with proper GLSL std140 alignment
        offset = 0
        
        for name, uniform in self.uniforms.items():
            # Align offset to the uniform's specific alignment requirement (std140 rules)
            alignment = uniform._get_alignment()
            offset = self._calculate_aligned_offset(offset, alignment)
            uniform.offset = offset
            offset += uniform.vk_size
        
        # Ensure the total buffer size meets minimum uniform buffer alignment
        uniform_buffer_alignment = self._get_uniform_buffer_alignment()
        self.total_size = self._calculate_aligned_offset(offset, uniform_buffer_alignment)
        
        # Create the Vulkan buffer
        self._vk_buffer = vk.vkCreateBuffer(
            self.app._vk_device,
            vk.VkBufferCreateInfo(
                size=self.total_size,
                usage=vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
            ),
            None
        )
        
        # Allocate memory for the buffer
        memory_req = vk.vkGetBufferMemoryRequirements(self.app._vk_device, self._vk_buffer)
        memory_type_index = self.app.get_memory_type_index(
            memory_req.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        
        self._vk_memory = vk.vkAllocateMemory(
            self.app._vk_device,
            vk.VkMemoryAllocateInfo(
                allocationSize=memory_req.size,
                memoryTypeIndex=memory_type_index
            ),
            None
        )
        
        # Bind buffer to memory
        vk.vkBindBufferMemory(self.app._vk_device, self._vk_buffer, self._vk_memory, 0)
        
        # Map memory persistently
        self.mapped_memory = vk.vkMapMemory(
            self.app._vk_device, 
            self._vk_memory, 0, 
            memory_req.size, 0
        )
        
        # Create descriptor set layout first (needed for pipeline creation)
        self.create_descriptor_set_layout()
        
        # Map each uniform to its location in the buffer
        for name, uniform in self.uniforms.items():
            uniform.map_to_location(self.mapped_memory, uniform.offset)
        
        # Create descriptor sets after buffer is ready
        self.create_descriptor_set(self.app._vk_descriptor_pool)
        
        print(f"UniformBuffer created: {self.total_size} bytes, {len(self.uniforms)} uniforms")
        for name, uniform in self.uniforms.items():
            print(f"  {name}: offset={uniform.offset}, size={uniform.vk_size}")
    
    def destroy(self):
        """Clean up Vulkan resources"""
        if self.mapped_memory is not None:
            vk.vkUnmapMemory(self.app._vk_device, self._vk_memory)
            self.mapped_memory = None
        
        if self._vk_memory is not None:
            vk.vkFreeMemory(self.app._vk_device, self._vk_memory, None)
            self._vk_memory = None
        
        if self._vk_buffer is not None:
            vk.vkDestroyBuffer(self.app._vk_device, self._vk_buffer, None)
            self._vk_buffer = None


class MyApp(App):

    def __init__(self):
        super().__init__('MyApp')

        triangle = np.array([
            [-0.8, -0.8, 1.0, 0.0, 0.0],
            [0.8, -0.8, 0.0, 1.0, 0.0],
            [0.0, 0.8, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Create uniform buffer with proper initialization
        self.uniforms = UniformBuffer(self)
        self.uniforms['camera'] = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
        self.uniforms['time'] = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
        self.uniforms.make_buffer()
        
        self.pass1 = Pass(self)
        # Now test with uniform shader
        self.mesh1 = Drawable(
            self, triangle, 
            './glsl/triangle.vert', './glsl/triangle.frag', 
            self.pass1, 
            vertex_attributes=['vec2', 'vec3'],
            uniforms=self.uniforms)
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 0.2

    def draw(self, command_buffer, swapchain_image):
        with self.pass1.start(command_buffer, swapchain_image):
            self.mesh1.draw(command_buffer, 0)

    def main_loop(self):
        # handle user input, update uniforms, etc.
        t = glfw.get_time()
        
        # Test uniform updates - this demonstrates the persistent mapping working!
        # Update time uniform
        self.uniforms['time'][0] = t
        
        # Animate camera matrix - translate X based on time
        self.uniforms['camera'][0, 3] = 0.2 * np.sin(t)  # X translation
        
        # FPS display
        if t - self.last_time > self.fps_interval:
            fps = (self.frame_count - self.last_count) / (t - self.last_time)
            self.last_count = self.frame_count
            self.last_time = t
            glfw.set_window_title(self.window, f'{self.title} {fps:.2f}fps - Time: {t:.1f}')


if __name__ == '__main__':
    app = MyApp()
    app.run()
