import vulkan as vk
import glfw
from .swapchain import Swapchain
from .frame import Frame
from threading import Thread


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
        self.framebuffer_resized = False

        self.init_vk()
        self.create_descriptor_pool()  # Create descriptor pool for uniform buffers
        self.swapchain = Swapchain(self, n_images)
        self.frames = tuple(Frame(self) for _ in range(n_frames))
         
    def init_vk(self):

        self.window = glfw.create_window(*self.size, self.title, None, None)
        # glfw.set_framebuffer_size_callback(self.window, self.framebuffer_resize_callback)

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
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
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
        try:
            frame.fence.wait()
            frame.fence.reset()
        except vk.VkTimeout:
            # If fence times out, reset it anyway and continue
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
            
            try:
                image, command_buffer = self.swapchain.get_next_image(frame)
            except (vk.VkError, vk.VkSuboptimalKhr) as e:
                if isinstance(e, vk.VkSuboptimalKhr):
                    self.recreate_swapchain()
                    continue
                elif hasattr(e, 'args') and len(e.args) > 0 and e.args[0] == vk.VK_ERROR_OUT_OF_DATE_KHR:
                    self.recreate_swapchain()
                    continue
                else:
                    raise e

            self.submit_commands(command_buffer, frame)

            try:
                self.swapchain.present_image(image, frame.render_finished_semaphore)
            except (vk.VkError, vk.VkSuboptimalKhr) as e:
                should_recreate = False
                
                if isinstance(e, vk.VkSuboptimalKhr):
                    should_recreate = True
                elif hasattr(e, 'args') and len(e.args) > 0 and e.args[0] in [vk.VK_ERROR_OUT_OF_DATE_KHR, vk.VK_SUBOPTIMAL_KHR]:
                    should_recreate = True
                elif self.framebuffer_resized:
                    should_recreate = True
                
                if should_recreate:
                    self.framebuffer_resized = False
                    self.recreate_swapchain()
                else:
                    raise e

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

    def framebuffer_resize_callback(self, window, width, height):
        self.framebuffer_resized = True

    def recreate_swapchain(self):
        w, h = glfw.get_framebuffer_size(self.window)
        while w == 0 or h == 0:
            w, h = glfw.get_framebuffer_size(self.window)
            glfw.wait_events()

        vk.vkDeviceWaitIdle(self._vk_device)

        self.cleanup_swapchain()

        # Recreate semaphores for all frames to avoid validation errors
        for frame in self.frames:
            frame.recreate_sync_objects()

        self.supported_surface_capabilities = vk.vkGetInstanceProcAddr(
            self._vk_instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR'
        )(self._vk_physical_device, self._vk_surface, None)
        self._vk_extent = vk.VkExtent2D(
            width=max(min(w, self.supported_surface_capabilities.maxImageExtent.width), self.supported_surface_capabilities.minImageExtent.width), 
            height=max(min(h, self.supported_surface_capabilities.maxImageExtent.height), self.supported_surface_capabilities.minImageExtent.height)
        )
        self._vk_viewport.width = w
        self._vk_viewport.height = h

        self.create_swapchain()
        self.record_draw_commands()

    def cleanup_swapchain(self):
        raise NotImplementedError()

    def create_swapchain(self):
        raise NotImplementedError()
