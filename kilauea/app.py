import vulkan as vk
import glfw
from .swapchain import Swapchain
from .frame import Frame
from threading import Thread
import subprocess
import os
import tempfile


class App:

    def __init__(
                self, title='Kilauea', size=(640, 480), n_frames=3, n_images=4, version=(1, 3, 0), 
                engine_name='Kilauea', device_preference=['discrete_gpu', 'integrated_gpu', 'virtual_gpu', 'cpu'],
                surface_format=vk.VK_FORMAT_B8G8R8A8_UNORM, color_space=vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR

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
        self.swapchain.create()
        self.frames = tuple(Frame(self) for _ in range(n_frames))
        
    def initialize_objects(self):
        """Initialize all registered objects after user's __init__ is complete"""
        for obj in self.swapchain.objects:
            obj.create()
         
    def init_vk(self):

        # Handle fullscreen mode
        if self.size == 'fullscreen':
            # Get primary monitor and its video mode
            primary_monitor = glfw.get_primary_monitor()
            video_mode = glfw.get_video_mode(primary_monitor)
            self.window = glfw.create_window(video_mode.size.width, video_mode.size.height, self.title, primary_monitor, None)
            print(f"Created fullscreen window: {video_mode.size.width}x{video_mode.size.height}")
        else:
            self.window = glfw.create_window(*self.size, self.title, None, None)
        
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_resize_callback)

        app_info = vk.VkApplicationInfo(
            pApplicationName=self.title,
            applicationVersion=self.version,
            pEngineName=self.engine_name,
            engineVersion=self.version,
            apiVersion=self.version
        )

        extensions = glfw.get_required_instance_extensions()
        extensions.append(vk.VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        
        # Add pipeline executable properties extension for advanced debugging
        if hasattr(vk, 'VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME'):
            extensions.append(vk.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)
        
        supported_extensions = [e.extensionName for e in vk.vkEnumerateInstanceExtensionProperties(None)]
        for e in extensions:
            if e not in supported_extensions:
                print(f'Warning: Extension {e} is not supported, skipping...')
        
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
            flags = vk.VK_DEBUG_REPORT_ERROR_BIT_EXT | vk.VK_DEBUG_REPORT_WARNING_BIT_EXT | vk.VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT | vk.VK_DEBUG_REPORT_INFORMATION_BIT_EXT | vk.VK_DEBUG_REPORT_DEBUG_BIT_EXT,
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
        
        # Debug: Print device properties and limitations
        self.print_device_properties()
        
        w, h = glfw.get_framebuffer_size(self.window)
        self.supported_surface_capabilities = vk.vkGetInstanceProcAddr(
            self._vk_instance, 'vkGetPhysicalDeviceSurfaceCapabilitiesKHR'
        )(self._vk_physical_device, self._vk_surface, None)
        
        # Store the window extent (full resolution) - this is also the swapchain extent
        if self.supported_surface_capabilities.currentExtent.width != 0xFFFFFFFF:
            self._vk_extent = self.supported_surface_capabilities.currentExtent
        else:
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
        
        # Check for pipeline executable properties extension
        device_ext_props = vk.vkEnumerateDeviceExtensionProperties(self._vk_physical_device, None)
        available_device_extensions = [ext.extensionName for ext in device_ext_props]
        
        if hasattr(vk, 'VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME'):
            if vk.VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME in available_device_extensions:
                device_extensions.append(vk.VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)
                print(f"Added VK_KHR_pipeline_executable_properties extension for advanced debugging")
            else:
                print(f"VK_KHR_pipeline_executable_properties not available on this device")

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

        print("\n--- INITIALIZATION ---")
        print(f"Framebuffer size: ({w}, {h})")
        print(f"Swapchain Extent: ({self._vk_extent.width}, {self._vk_extent.height})")
        print(f"Initial Viewport: (x={self._vk_viewport.x}, y={self._vk_viewport.y}, w={self._vk_viewport.width}, h={self._vk_viewport.height})")

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

    def print_device_properties(self):
        """Debug method to print VkPhysicalDeviceProperties and device limitations"""
        print("\n" + "="*80)
        print("VULKAN PHYSICAL DEVICE PROPERTIES AND LIMITATIONS")
        print("="*80)
        
        # Get basic device properties
        props = vk.vkGetPhysicalDeviceProperties(self._vk_physical_device)
        
        # Device type mapping for readable output
        device_types = {
            vk.VK_PHYSICAL_DEVICE_TYPE_OTHER: "Other",
            vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: "Integrated GPU",
            vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: "Discrete GPU",
            vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: "Virtual GPU",
            vk.VK_PHYSICAL_DEVICE_TYPE_CPU: "CPU"
        }
        
        print(f"\nDevice Name: {props.deviceName}")
        print(f"Device Type: {device_types.get(props.deviceType, 'Unknown')}")
        print(f"Vendor ID: 0x{props.vendorID:04X}")
        print(f"Device ID: 0x{props.deviceID:04X}")
        print(f"Driver Version: {props.driverVersion}")
        print(f"API Version: {vk.VK_VERSION_MAJOR(props.apiVersion)}.{vk.VK_VERSION_MINOR(props.apiVersion)}.{vk.VK_VERSION_PATCH(props.apiVersion)}")
        
        # Device Limitations (VkPhysicalDeviceLimits)
        limits = props.limits
        print(f"\n--- DEVICE LIMITATIONS ---")
        print(f"Max Image Dimension 1D: {limits.maxImageDimension1D}")
        print(f"Max Image Dimension 2D: {limits.maxImageDimension2D}")
        print(f"Max Image Dimension 3D: {limits.maxImageDimension3D}")
        print(f"Max Image Dimension Cube: {limits.maxImageDimensionCube}")
        print(f"Max Image Array Layers: {limits.maxImageArrayLayers}")
        print(f"Max Texel Buffer Elements: {limits.maxTexelBufferElements}")
        print(f"Max Uniform Buffer Range: {limits.maxUniformBufferRange}")
        print(f"Max Storage Buffer Range: {limits.maxStorageBufferRange}")
        print(f"Max Push Constants Size: {limits.maxPushConstantsSize}")
        print(f"Max Memory Allocation Count: {limits.maxMemoryAllocationCount}")
        print(f"Max Sampler Allocation Count: {limits.maxSamplerAllocationCount}")
        print(f"Buffer Image Granularity: {limits.bufferImageGranularity}")
        print(f"Sparse Address Space Size: {limits.sparseAddressSpaceSize}")
        print(f"Max Bound Descriptor Sets: {limits.maxBoundDescriptorSets}")
        print(f"Max Per Stage Descriptor Samplers: {limits.maxPerStageDescriptorSamplers}")
        print(f"Max Per Stage Descriptor Uniform Buffers: {limits.maxPerStageDescriptorUniformBuffers}")
        print(f"Max Per Stage Descriptor Storage Buffers: {limits.maxPerStageDescriptorStorageBuffers}")
        print(f"Max Per Stage Descriptor Sampled Images: {limits.maxPerStageDescriptorSampledImages}")
        print(f"Max Per Stage Descriptor Storage Images: {limits.maxPerStageDescriptorStorageImages}")
        print(f"Max Per Stage Descriptor Input Attachments: {limits.maxPerStageDescriptorInputAttachments}")
        print(f"Max Per Stage Resources: {limits.maxPerStageResources}")
        print(f"Max Descriptor Set Samplers: {limits.maxDescriptorSetSamplers}")
        print(f"Max Descriptor Set Uniform Buffers: {limits.maxDescriptorSetUniformBuffers}")
        print(f"Max Descriptor Set Uniform Buffers Dynamic: {limits.maxDescriptorSetUniformBuffersDynamic}")
        print(f"Max Descriptor Set Storage Buffers: {limits.maxDescriptorSetStorageBuffers}")
        print(f"Max Descriptor Set Storage Buffers Dynamic: {limits.maxDescriptorSetStorageBuffersDynamic}")
        print(f"Max Descriptor Set Sampled Images: {limits.maxDescriptorSetSampledImages}")
        print(f"Max Descriptor Set Storage Images: {limits.maxDescriptorSetStorageImages}")
        print(f"Max Descriptor Set Input Attachments: {limits.maxDescriptorSetInputAttachments}")
        print(f"Max Vertex Input Attributes: {limits.maxVertexInputAttributes}")
        print(f"Max Vertex Input Bindings: {limits.maxVertexInputBindings}")
        print(f"Max Vertex Input Attribute Offset: {limits.maxVertexInputAttributeOffset}")
        print(f"Max Vertex Input Binding Stride: {limits.maxVertexInputBindingStride}")
        print(f"Max Vertex Output Components: {limits.maxVertexOutputComponents}")
        print(f"Max Tessellation Generation Level: {limits.maxTessellationGenerationLevel}")
        print(f"Max Tessellation Patch Size: {limits.maxTessellationPatchSize}")
        print(f"Max Tessellation Control Per Vertex Input Components: {limits.maxTessellationControlPerVertexInputComponents}")
        print(f"Max Tessellation Control Per Vertex Output Components: {limits.maxTessellationControlPerVertexOutputComponents}")
        print(f"Max Tessellation Control Per Patch Output Components: {limits.maxTessellationControlPerPatchOutputComponents}")
        print(f"Max Tessellation Control Total Output Components: {limits.maxTessellationControlTotalOutputComponents}")
        print(f"Max Tessellation Evaluation Input Components: {limits.maxTessellationEvaluationInputComponents}")
        print(f"Max Tessellation Evaluation Output Components: {limits.maxTessellationEvaluationOutputComponents}")
        print(f"Max Geometry Shader Invocations: {limits.maxGeometryShaderInvocations}")
        print(f"Max Geometry Input Components: {limits.maxGeometryInputComponents}")
        print(f"Max Geometry Output Components: {limits.maxGeometryOutputComponents}")
        print(f"Max Geometry Output Vertices: {limits.maxGeometryOutputVertices}")
        print(f"Max Geometry Total Output Components: {limits.maxGeometryTotalOutputComponents}")
        print(f"Max Fragment Input Components: {limits.maxFragmentInputComponents}")
        print(f"Max Fragment Output Attachments: {limits.maxFragmentOutputAttachments}")
        print(f"Max Fragment Dual Src Attachments: {limits.maxFragmentDualSrcAttachments}")
        print(f"Max Fragment Combined Output Resources: {limits.maxFragmentCombinedOutputResources}")
        print(f"Max Compute Shared Memory Size: {limits.maxComputeSharedMemorySize}")
        print(f"Max Compute Work Group Count: {limits.maxComputeWorkGroupCount}")
        print(f"Max Compute Work Group Invocations: {limits.maxComputeWorkGroupInvocations}")
        print(f"Max Compute Work Group Size: {limits.maxComputeWorkGroupSize}")
        print(f"Sub Pixel Precision Bits: {limits.subPixelPrecisionBits}")
        print(f"Sub Texel Precision Bits: {limits.subTexelPrecisionBits}")
        print(f"Mipmap Precision Bits: {limits.mipmapPrecisionBits}")
        print(f"Max Draw Indexed Index Value: {limits.maxDrawIndexedIndexValue}")
        print(f"Max Draw Indirect Count: {limits.maxDrawIndirectCount}")
        print(f"Max Sampler Lod Bias: {limits.maxSamplerLodBias}")
        print(f"Max Sampler Anisotropy: {limits.maxSamplerAnisotropy}")
        print(f"Max Viewports: {limits.maxViewports}")
        print(f"Max Viewport Dimensions: {limits.maxViewportDimensions}")
        print(f"Viewport Bounds Range: {limits.viewportBoundsRange}")
        print(f"Viewport Sub Pixel Bits: {limits.viewportSubPixelBits}")
        print(f"Min Memory Map Alignment: {limits.minMemoryMapAlignment}")
        print(f"Min Texel Buffer Offset Alignment: {limits.minTexelBufferOffsetAlignment}")
        print(f"Min Uniform Buffer Offset Alignment: {limits.minUniformBufferOffsetAlignment}")
        print(f"Min Storage Buffer Offset Alignment: {limits.minStorageBufferOffsetAlignment}")
        print(f"Min Texel Offset: {limits.minTexelOffset}")
        print(f"Max Texel Offset: {limits.maxTexelOffset}")
        print(f"Min Texel Gather Offset: {limits.minTexelGatherOffset}")
        print(f"Max Texel Gather Offset: {limits.maxTexelGatherOffset}")
        print(f"Min Interpolation Offset: {limits.minInterpolationOffset}")
        print(f"Max Interpolation Offset: {limits.maxInterpolationOffset}")
        print(f"Sub Pixel Interpolation Offset Bits: {limits.subPixelInterpolationOffsetBits}")
        print(f"Max Framebuffer Width: {limits.maxFramebufferWidth}")
        print(f"Max Framebuffer Height: {limits.maxFramebufferHeight}")
        print(f"Max Framebuffer Layers: {limits.maxFramebufferLayers}")
        print(f"Framebuffer Color Sample Counts: {limits.framebufferColorSampleCounts}")
        print(f"Framebuffer Depth Sample Counts: {limits.framebufferDepthSampleCounts}")
        print(f"Framebuffer Stencil Sample Counts: {limits.framebufferStencilSampleCounts}")
        print(f"Framebuffer No Attachments Sample Counts: {limits.framebufferNoAttachmentsSampleCounts}")
        print(f"Max Color Attachments: {limits.maxColorAttachments}")
        print(f"Sampled Image Color Sample Counts: {limits.sampledImageColorSampleCounts}")
        print(f"Sampled Image Integer Sample Counts: {limits.sampledImageIntegerSampleCounts}")
        print(f"Sampled Image Depth Sample Counts: {limits.sampledImageDepthSampleCounts}")
        print(f"Sampled Image Stencil Sample Counts: {limits.sampledImageStencilSampleCounts}")
        print(f"Storage Image Sample Counts: {limits.storageImageSampleCounts}")
        print(f"Max Sample Mask Words: {limits.maxSampleMaskWords}")
        print(f"Timestamp Compute And Graphics: {limits.timestampComputeAndGraphics}")
        print(f"Timestamp Period: {limits.timestampPeriod}")
        print(f"Max Clip Distances: {limits.maxClipDistances}")
        print(f"Max Cull Distances: {limits.maxCullDistances}")
        print(f"Max Combined Clip And Cull Distances: {limits.maxCombinedClipAndCullDistances}")
        print(f"Discrete Queue Priorities: {limits.discreteQueuePriorities}")
        print(f"Point Size Range: {limits.pointSizeRange}")
        print(f"Line Width Range: {limits.lineWidthRange}")
        print(f"Point Size Granularity: {limits.pointSizeGranularity}")
        print(f"Line Width Granularity: {limits.lineWidthGranularity}")
        print(f"Strict Lines: {limits.strictLines}")
        print(f"Standard Sample Locations: {limits.standardSampleLocations}")
        print(f"Optimal Buffer Copy Offset Alignment: {limits.optimalBufferCopyOffsetAlignment}")
        print(f"Optimal Buffer Copy Row Pitch Alignment: {limits.optimalBufferCopyRowPitchAlignment}")
        print(f"Non Coherent Atom Size: {limits.nonCoherentAtomSize}")
        
        # Try to get extended properties using vkGetPhysicalDeviceProperties2
        try:
            self.print_device_properties2()
        except Exception as e:
            print(f"\nNote: Could not retrieve extended properties via vkGetPhysicalDeviceProperties2: {e}")
        
        print("="*80)
        print()

    def print_device_properties2(self):
        """Print extended device properties using vkGetPhysicalDeviceProperties2"""
        print(f"\n--- EXTENDED PROPERTIES (vkGetPhysicalDeviceProperties2) ---")
        
        # Check if vkGetPhysicalDeviceProperties2 is available
        try:
            get_props2_func = vk.vkGetInstanceProcAddr(self._vk_instance, 'vkGetPhysicalDeviceProperties2')
            if not get_props2_func:
                print("vkGetPhysicalDeviceProperties2 not available in this Vulkan instance")
                return
            
            # Create the properties2 structure
            props2 = vk.VkPhysicalDeviceProperties2(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                pNext=None
            )
            
            # Call vkGetPhysicalDeviceProperties2
            get_props2_func(self._vk_physical_device, props2)
            
            # The basic properties are the same as vkGetPhysicalDeviceProperties
            # but this shows we can successfully call the extended version
            print(f"Extended Properties Retrieved Successfully")
            print(f"Structure Type: {props2.sType}")
            print(f"Device Name (via Props2): {props2.properties.deviceName}")
            print(f"API Version (via Props2): {vk.VK_VERSION_MAJOR(props2.properties.apiVersion)}.{vk.VK_VERSION_MINOR(props2.properties.apiVersion)}.{vk.VK_VERSION_PATCH(props2.properties.apiVersion)}")
            
            # Note: To get truly extended properties, you would chain additional structures
            # like VkPhysicalDeviceVulkan11Properties, VkPhysicalDeviceVulkan12Properties, etc.
            # in the pNext chain, but this demonstrates the basic usage
            
        except Exception as e:
            print(f"vkGetPhysicalDeviceProperties2 failed: {str(e)}")
            print("This may be due to Vulkan version compatibility or Python binding limitations")

    def debug_shader_spirv(self, shader_path, stage_name="unknown"):
        """Debug SPIR-V output for a shader file using glslc and spirv-dis"""
        print(f"\n--- SPIR-V DEBUGGING FOR {shader_path} ({stage_name}) ---")
        
        if not os.path.exists(shader_path):
            print(f"Shader file {shader_path} not found")
            return
        
        try:
            # Create temporary files for SPIR-V compilation
            with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as spv_file:
                spv_path = spv_file.name
            
            # Compile GLSL to SPIR-V using glslc
            compile_cmd = ['glslc', shader_path, '-o', spv_path, '-O']
            try:
                result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    print(f"glslc compilation failed: {result.stderr}")
                    return
                print(f"Successfully compiled {shader_path} to SPIR-V")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"glslc not available or failed: {e}")
                print("Install Vulkan SDK to get glslc compiler for SPIR-V debugging")
                return
            
            # Disassemble SPIR-V using spirv-dis
            disasm_cmd = ['spirv-dis', spv_path]
            try:
                result = subprocess.run(disasm_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"\nSPIR-V Disassembly for {shader_path}:")
                    print("-" * 60)
                    lines = result.stdout.split('\n')
                    # Show first 50 lines to avoid overwhelming output
                    for i, line in enumerate(lines[:50]):
                        print(f"{i+1:3d}: {line}")
                    if len(lines) > 50:
                        print(f"... ({len(lines) - 50} more lines)")
                    print("-" * 60)
                    
                    # Analyze for potential issues
                    self.analyze_spirv_output(result.stdout, shader_path)
                else:
                    print(f"spirv-dis failed: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"spirv-dis not available: {e}")
                print("Install Vulkan SDK to get spirv-dis for SPIR-V disassembly")
            
            # Clean up temporary file
            try:
                os.unlink(spv_path)
            except:
                pass
                
        except Exception as e:
            print(f"SPIR-V debugging failed: {e}")

    def analyze_spirv_output(self, spirv_text, shader_path):
        """Analyze SPIR-V disassembly for potential performance issues"""
        print(f"\n--- SPIR-V ANALYSIS FOR {shader_path} ---")
        
        lines = spirv_text.split('\n')
        
        # Count instructions
        instruction_count = len([line for line in lines if line.strip() and not line.strip().startswith(';')])
        print(f"Total SPIR-V Instructions: {instruction_count}")
        
        # Look for loops (potential unrolling)
        loop_count = len([line for line in lines if 'OpLoopMerge' in line])
        if loop_count > 0:
            print(f"Loops detected: {loop_count} (check for unrolling opportunities)")
        
        # Look for branches
        branch_count = len([line for line in lines if any(op in line for op in ['OpBranch', 'OpBranchConditional'])])
        if branch_count > 10:
            print(f"High branch count: {branch_count} (may impact performance)")
        
        # Look for texture operations
        texture_ops = len([line for line in lines if any(op in line for op in ['OpImageSample', 'OpImageFetch', 'OpImageRead'])])
        if texture_ops > 0:
            print(f"Texture operations: {texture_ops}")
        
        # Look for arithmetic operations
        math_ops = len([line for line in lines if any(op in line for op in ['OpFAdd', 'OpFMul', 'OpFDiv', 'OpFSub'])])
        if math_ops > 0:
            print(f"Floating-point arithmetic operations: {math_ops}")
        
        # Look for potential optimization issues
        if 'OpFDiv' in spirv_text:
            div_count = len([line for line in lines if 'OpFDiv' in line])
            print(f"Warning: {div_count} division operations detected (consider using multiplication by reciprocal)")
        
        if 'OpPow' in spirv_text:
            print("Warning: Power operations detected (expensive, consider alternatives)")

    def debug_pipeline_executable_properties(self, pipeline):
        """Debug pipeline using VK_KHR_pipeline_executable_properties"""
        print(f"\n--- PIPELINE EXECUTABLE PROPERTIES DEBUG ---")
        
        try:
            # Check if the extension functions are available
            get_pipeline_executable_properties_func = vk.vkGetDeviceProcAddr(
                self._vk_device, 'vkGetPipelineExecutablePropertiesKHR'
            )
            get_pipeline_executable_statistics_func = vk.vkGetDeviceProcAddr(
                self._vk_device, 'vkGetPipelineExecutableStatisticsKHR'
            )
            
            if not get_pipeline_executable_properties_func:
                print("VK_KHR_pipeline_executable_properties functions not available")
                return
            
            # Get pipeline executable properties
            pipeline_info = vk.VkPipelineInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
                pipeline=pipeline
            )
            
            # Get executable count
            executable_count = vk.ffi.new('uint32_t *')
            get_pipeline_executable_properties_func(
                self._vk_device, pipeline_info, executable_count, vk.ffi.NULL
            )
            
            if executable_count[0] == 0:
                print("No pipeline executables found")
                return
            
            print(f"Found {executable_count[0]} pipeline executable(s)")
            
            # Get executable properties
            properties = vk.ffi.new(f'VkPipelineExecutablePropertiesKHR[{executable_count[0]}]')
            for i in range(executable_count[0]):
                properties[i].sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR
            
            get_pipeline_executable_properties_func(
                self._vk_device, pipeline_info, executable_count, properties
            )
            
            # Print properties for each executable
            for i in range(executable_count[0]):
                prop = properties[i]
                print(f"\nExecutable {i}:")
                print(f"  Name: {vk.ffi.string(prop.name).decode()}")
                print(f"  Description: {vk.ffi.string(prop.description).decode()}")
                print(f"  Stages: {prop.stages}")
                print(f"  Subgroup Size: {prop.subgroupSize}")
                
                # Get statistics for this executable
                if get_pipeline_executable_statistics_func:
                    self.debug_pipeline_executable_statistics(
                        get_pipeline_executable_statistics_func, pipeline, i
                    )
                    
        except Exception as e:
            print(f"Pipeline executable properties debugging failed: {e}")
            print("This may be due to driver limitations or extension availability")

    def debug_pipeline_executable_statistics(self, stats_func, pipeline, executable_index):
        """Get and print pipeline executable statistics"""
        try:
            pipeline_executable_info = vk.VkPipelineExecutableInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
                pipeline=pipeline,
                executableIndex=executable_index
            )
            
            # Get statistics count
            stats_count = vk.ffi.new('uint32_t *')
            stats_func(
                self._vk_device, pipeline_executable_info, stats_count, vk.ffi.NULL
            )
            
            if stats_count[0] == 0:
                print("    No statistics available")
                return
            
            # Get statistics
            statistics = vk.ffi.new(f'VkPipelineExecutableStatisticKHR[{stats_count[0]}]')
            for i in range(stats_count[0]):
                statistics[i].sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR
            
            stats_func(
                self._vk_device, pipeline_executable_info, stats_count, statistics
            )
            
            print(f"  Statistics ({stats_count[0]} entries):")
            for i in range(stats_count[0]):
                stat = statistics[i]
                name = vk.ffi.string(stat.name).decode()
                description = vk.ffi.string(stat.description).decode()
                
                # Format value based on type
                if stat.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                    value = "True" if stat.value.b32 else "False"
                elif stat.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                    value = str(stat.value.i64)
                elif stat.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                    value = str(stat.value.u64)
                elif stat.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                    value = f"{stat.value.f64:.6f}"
                else:
                    value = "Unknown format"
                
                print(f"    {name}: {value}")
                if description:
                    print(f"      ({description})")
                    
        except Exception as e:
            print(f"    Failed to get executable statistics: {e}")

    def debug_graphics_pipeline_creation(self, shader_stages, vertex_input_state, input_assembly_state, 
                                       viewport_state, rasterization_state, multisample_state, 
                                       color_blend_state, render_pass, pipeline_layout, subpass=0):
        """Debug graphics pipeline creation issues"""
        print(f"\n--- GRAPHICS PIPELINE CREATION DEBUG ---")
        
        # Validate shader stages
        print(f"Shader Stages ({len(shader_stages)} stages):")
        for i, stage in enumerate(shader_stages):
            stage_names = {
                vk.VK_SHADER_STAGE_VERTEX_BIT: "Vertex",
                vk.VK_SHADER_STAGE_FRAGMENT_BIT: "Fragment",
                vk.VK_SHADER_STAGE_GEOMETRY_BIT: "Geometry",
                vk.VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: "Tessellation Control",
                vk.VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: "Tessellation Evaluation",
                vk.VK_SHADER_STAGE_COMPUTE_BIT: "Compute"
            }
            stage_name = stage_names.get(stage.stage, f"Unknown({stage.stage})")
            print(f"  Stage {i}: {stage_name}")
            print(f"    Module: {stage.module}")
            print(f"    Entry Point: {stage.pName}")
            
            # Validate shader module
            if not stage.module:
                print(f"    ERROR: Shader module is NULL!")
            
            # Check entry point - convert to string for proper display
            if not stage.pName:
                print(f"    ERROR: Entry point is NULL!")
            else:
                try:
                    entry_point_str = vk.ffi.string(stage.pName).decode('utf-8')
                    print(f"    Entry Point: '{entry_point_str}'")
                    if entry_point_str != 'main':
                        print(f"    WARNING: Entry point is not 'main': '{entry_point_str}'")
                except:
                    print(f"    Entry Point: {stage.pName} (could not decode)")
        
        # Validate vertex input state
        if vertex_input_state:
            print(f"\nVertex Input State:")
            print(f"  Binding Descriptions: {vertex_input_state.vertexBindingDescriptionCount}")
            print(f"  Attribute Descriptions: {vertex_input_state.vertexAttributeDescriptionCount}")
            
            # Check for common vertex input issues
            if vertex_input_state.vertexBindingDescriptionCount == 0:
                print(f"    WARNING: No vertex binding descriptions")
            if vertex_input_state.vertexAttributeDescriptionCount == 0:
                print(f"    WARNING: No vertex attribute descriptions")
        else:
            print(f"\nVertex Input State: NULL")
        
        # Validate input assembly
        if input_assembly_state:
            topology_names = {
                vk.VK_PRIMITIVE_TOPOLOGY_POINT_LIST: "Point List",
                vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST: "Line List",
                vk.VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: "Line Strip",
                vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: "Triangle List",
                vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: "Triangle Strip",
                vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: "Triangle Fan"
            }
            topology_name = topology_names.get(input_assembly_state.topology, f"Unknown({input_assembly_state.topology})")
            print(f"\nInput Assembly State:")
            print(f"  Topology: {topology_name}")
            print(f"  Primitive Restart: {input_assembly_state.primitiveRestartEnable}")
        
        # Validate viewport state
        if viewport_state:
            print(f"\nViewport State:")
            print(f"  Viewport Count: {viewport_state.viewportCount}")
            print(f"  Scissor Count: {viewport_state.scissorCount}")
            
            if viewport_state.viewportCount == 0:
                print(f"    ERROR: Viewport count is 0!")
            if viewport_state.scissorCount == 0:
                print(f"    ERROR: Scissor count is 0!")
        
        # Validate rasterization state
        if rasterization_state:
            print(f"\nRasterization State:")
            print(f"  Depth Clamp: {rasterization_state.depthClampEnable}")
            print(f"  Rasterizer Discard: {rasterization_state.rasterizerDiscardEnable}")
            print(f"  Polygon Mode: {rasterization_state.polygonMode}")
            print(f"  Cull Mode: {rasterization_state.cullMode}")
            print(f"  Front Face: {rasterization_state.frontFace}")
            
            if rasterization_state.rasterizerDiscardEnable:
                print(f"    WARNING: Rasterizer discard is enabled!")
        
        # Validate multisample state
        if multisample_state:
            print(f"\nMultisample State:")
            print(f"  Sample Count: {multisample_state.rasterizationSamples}")
            print(f"  Sample Shading: {multisample_state.sampleShadingEnable}")
        
        # Validate color blend state
        if color_blend_state:
            print(f"\nColor Blend State:")
            print(f"  Logic Op Enable: {color_blend_state.logicOpEnable}")
            print(f"  Attachment Count: {color_blend_state.attachmentCount}")
            
            if color_blend_state.attachmentCount == 0:
                print(f"    WARNING: No color blend attachments!")
        
        # Validate render pass
        print(f"\nRender Pass: {render_pass}")
        if not render_pass:
            print(f"    ERROR: Render pass is NULL!")
        
        # Validate pipeline layout
        print(f"Pipeline Layout: {pipeline_layout}")
        if not pipeline_layout:
            print(f"    ERROR: Pipeline layout is NULL!")
        
        print(f"Subpass: {subpass}")
        
        # Try to create the pipeline with detailed error reporting
        print(f"\n--- ATTEMPTING PIPELINE CREATION ---")
        
        # Clear previous validation messages to capture only pipeline creation messages
        self.clear_validation_messages()
        
        try:
            pipeline_create_info = vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                stageCount=len(shader_stages),
                pStages=shader_stages,
                pVertexInputState=vertex_input_state,
                pInputAssemblyState=input_assembly_state,
                pViewportState=viewport_state,
                pRasterizationState=rasterization_state,
                pMultisampleState=multisample_state,
                pColorBlendState=color_blend_state,
                layout=pipeline_layout,
                renderPass=render_pass,
                subpass=subpass
            )
            
            pipeline = vk.vkCreateGraphicsPipelines(
                self._vk_device, vk.VK_NULL_HANDLE, 1, [pipeline_create_info], None
            )
            
            # Get validation messages that occurred during pipeline creation
            validation_messages = self.get_recent_validation_messages(20)
            
            # Detailed analysis of the pipeline result
            print(f"Pipeline result analysis:")
            print(f"  Type: {type(pipeline)}")
            print(f"  Value: {pipeline}")
            print(f"  Length: {len(pipeline) if pipeline else 'N/A'}")
            
            if pipeline and len(pipeline) > 0:
                print(f"  First element: {pipeline[0]}")
                print(f"  First element type: {type(pipeline[0])}")
                print(f"  First element str: {str(pipeline[0])}")
                
                # Check if the pipeline handle is actually valid
                pipeline_handle = pipeline[0]
                if pipeline_handle and str(pipeline_handle) != 'NULL' and not str(pipeline_handle).find('NULL') != -1:
                    print(f"SUCCESS: Pipeline created successfully: {pipeline_handle}")
                    
                    # Show any warnings that occurred during successful creation
                    if validation_messages:
                        print(f"\nValidation messages during pipeline creation:")
                        for i, msg in enumerate(validation_messages):
                            print(f"  {i+1}. {msg['message']}")
                            if msg['details']:
                                print(f"     Details: {msg['details']}")
                    
                    return pipeline_handle
                else:
                    print(f"FAILURE: Pipeline handle is NULL: {pipeline_handle}")
            else:
                print(f"FAILURE: vkCreateGraphicsPipelines returned None or empty result")
            
            # Show validation messages that explain WHY it failed
            if validation_messages:
                print(f"\n*** VALIDATION LAYER MESSAGES (THE REAL REASON FOR FAILURE) ***")
                for i, msg in enumerate(validation_messages):
                    severity = msg.get('severity', 'unknown')
                    print(f"  {i+1}. [{severity}] {msg['message']}")
                    if msg['details']:
                        print(f"     Details: {msg['details']}")
                print(f"*** END VALIDATION MESSAGES ***")
            else:
                print(f"\nNo validation messages captured")
                print(f"This could mean:")
                print(f"1. Validation layers are not working properly")
                print(f"2. The failure is not generating validation messages")
                print(f"3. The pipeline creation is actually succeeding but we're misdetecting it")
            
            # Additional debugging suggestions
            print(f"\nDEBUGGING SUGGESTIONS:")
            print(f"1. Review the validation layer messages above for the exact error")
            print(f"2. Verify shader compilation succeeded")
            print(f"3. Ensure vertex input layout matches shader expectations")
            print(f"4. Verify render pass compatibility")
            print(f"5. Check pipeline layout descriptor set layouts")
            
            return None
                
        except Exception as e:
            print(f"EXCEPTION during pipeline creation: {e}")
            print(f"Exception type: {type(e)}")
            
            # Show validation messages that occurred before the exception
            validation_messages = self.get_recent_validation_messages(20)
            if validation_messages:
                print(f"\n*** VALIDATION MESSAGES BEFORE EXCEPTION ***")
                for i, msg in enumerate(validation_messages):
                    severity = msg.get('severity', 'unknown')
                    print(f"  {i+1}. [{severity}] {msg['message']}")
                    if msg['details']:
                        print(f"     Details: {msg['details']}")
                print(f"*** END VALIDATION MESSAGES ***")
            
            # Try to get more specific error information
            if hasattr(e, 'args') and len(e.args) > 0:
                print(f"Error code: {e.args[0]}")
                
                # Common Vulkan error codes
                error_codes = {
                    -1: "VK_ERROR_OUT_OF_HOST_MEMORY",
                    -2: "VK_ERROR_OUT_OF_DEVICE_MEMORY", 
                    -3: "VK_ERROR_INITIALIZATION_FAILED",
                    -4: "VK_ERROR_DEVICE_LOST",
                    -5: "VK_ERROR_MEMORY_MAP_FAILED",
                    -6: "VK_ERROR_LAYER_NOT_PRESENT",
                    -7: "VK_ERROR_EXTENSION_NOT_PRESENT",
                    -8: "VK_ERROR_FEATURE_NOT_PRESENT",
                    -9: "VK_ERROR_INCOMPATIBLE_DRIVER",
                    -10: "VK_ERROR_TOO_MANY_OBJECTS",
                    -11: "VK_ERROR_FORMAT_NOT_SUPPORTED",
                    -12: "VK_ERROR_FRAGMENTED_POOL"
                }
                
                if e.args[0] in error_codes:
                    print(f"Error meaning: {error_codes[e.args[0]]}")
            
            return None

    def debug_shader_module_creation(self, spirv_code, stage_name="unknown"):
        """Debug shader module creation"""
        print(f"\n--- SHADER MODULE CREATION DEBUG ({stage_name}) ---")
        
        if not spirv_code:
            print(f"ERROR: SPIR-V code is None or empty")
            return None
        
        print(f"SPIR-V code size: {len(spirv_code)} bytes")
        
        # Check SPIR-V magic number
        if len(spirv_code) >= 4:
            magic = int.from_bytes(spirv_code[:4], byteorder='little')
            expected_magic = 0x07230203
            print(f"SPIR-V magic number: 0x{magic:08X}")
            if magic != expected_magic:
                print(f"ERROR: Invalid SPIR-V magic number! Expected: 0x{expected_magic:08X}")
                return None
            else:
                print(f"SPIR-V magic number is valid")
        else:
            print(f"ERROR: SPIR-V code too short to contain magic number")
            return None
        
        # Check alignment
        if len(spirv_code) % 4 != 0:
            print(f"ERROR: SPIR-V code size is not 4-byte aligned")
            return None
        
        try:
            create_info = vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(spirv_code),
                pCode=spirv_code
            )
            
            shader_module = vk.vkCreateShaderModule(self._vk_device, create_info, None)
            
            if shader_module:
                print(f"SUCCESS: Shader module created: {shader_module}")
                return shader_module
            else:
                print(f"FAILURE: vkCreateShaderModule returned None")
                return None
                
        except Exception as e:
            print(f"EXCEPTION during shader module creation: {e}")
            print(f"Exception type: {type(e)}")
            return None

    # Class variable to store validation messages
    _validation_messages = []
    
    @staticmethod
    def debug_callback(*args):
        try:
            # Extract message and details more carefully
            message_text = args[5] if len(args) > 5 else "Unknown message"
            details_text = args[6] if len(args) > 6 else "No details"
            severity_flags = args[1] if len(args) > 1 else 0
            
            # Convert severity flags to readable text
            severity = "unknown"
            if severity_flags & vk.VK_DEBUG_REPORT_ERROR_BIT_EXT:
                severity = "error"
            elif severity_flags & vk.VK_DEBUG_REPORT_WARNING_BIT_EXT:
                severity = "warning"
            elif severity_flags & vk.VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
                severity = "performance"
            elif severity_flags & vk.VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
                severity = "info"
            elif severity_flags & vk.VK_DEBUG_REPORT_DEBUG_BIT_EXT:
                severity = "debug"
            
            print(f"Validation [{severity}]: {message_text}")
            if details_text and details_text != "No details":
                print(f"Details: {details_text}")
            
            # Store validation messages for pipeline debugging
            App._validation_messages.append({
                'message': str(message_text),
                'details': str(details_text),
                'severity': severity,
                'flags': severity_flags
            })
            
            # Keep only last 50 messages to avoid memory issues
            if len(App._validation_messages) > 50:
                App._validation_messages = App._validation_messages[-50:]
                
        except Exception as e:
            print(f"Error in debug callback: {e}")
            print(f"Args: {args}")
        
        # Don't raise exception to see more details
        return 0
    
    def clear_validation_messages(self):
        """Clear stored validation messages"""
        App._validation_messages.clear()
    
    def get_recent_validation_messages(self, count=10):
        """Get recent validation messages"""
        return App._validation_messages[-count:] if App._validation_messages else []

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
                is_out_of_date = isinstance(e, vk.VkSuboptimalKhr) or \
                                 (hasattr(e, 'args') and len(e.args) > 0 and e.args[0] == vk.VK_ERROR_OUT_OF_DATE_KHR)
                if is_out_of_date:
                    self.recreate_swapchain()
                    continue
                else:
                    raise e

            self.submit_commands(command_buffer, frame)

            present_failed = False
            try:
                self.swapchain.present_image(image, frame.render_finished_semaphore)
            except (vk.VkError, vk.VkSuboptimalKhr) as e:
                is_out_of_date = isinstance(e, vk.VkSuboptimalKhr) or \
                                 (hasattr(e, 'args') and len(e.args) > 0 and e.args[0] in [vk.VK_ERROR_OUT_OF_DATE_KHR, vk.VK_SUBOPTIMAL_KHR])
                if is_out_of_date:
                    present_failed = True
                else:
                    raise e
            
            if present_failed or self.framebuffer_resized:
                self.framebuffer_resized = False
                self.recreate_swapchain()

            self.frame_count += 1
    
    def submit_commands(self, command_buffer, frame):
        vk.vkQueueSubmit(self._vk_queue, 1, vk.VkSubmitInfo(
                waitSemaphoreCount=1, pWaitSemaphores=[frame.image_available_semaphore._vk_semaphore],
                pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,],
                commandBufferCount=1, pCommandBuffers=[command_buffer._vk_command_buffer],
                signalSemaphoreCount=1, pSignalSemaphores=[frame.render_finished_semaphore._vk_semaphore]
            ), fence=frame.fence._vk_fence)

    def run(self):
        # Initialize all registered objects after user's __init__ is complete
        self.initialize_objects()

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

        # Update swapchain extent to match new window size
        if self.supported_surface_capabilities.currentExtent.width != 0xFFFFFFFF:
            self._vk_extent = self.supported_surface_capabilities.currentExtent
        else:
            self._vk_extent = vk.VkExtent2D(
                width=max(min(w, self.supported_surface_capabilities.maxImageExtent.width), self.supported_surface_capabilities.minImageExtent.width), 
                height=max(min(h, self.supported_surface_capabilities.maxImageExtent.height), self.supported_surface_capabilities.minImageExtent.height)
            )
        
        self._vk_viewport.width = self._vk_extent.width
        self._vk_viewport.height = self._vk_extent.height

        print("\n--- RECREATE SWAPCHAIN ---")
        print(f"Framebuffer size: ({w}, {h})")
        print(f"New Swapchain Extent: ({self._vk_extent.width}, {self._vk_extent.height})")
        print(f"New Viewport: (x={self._vk_viewport.x}, y={self._vk_viewport.y}, w={self._vk_viewport.width}, h={self._vk_viewport.height})")

        self.create_swapchain()
        self.record_draw_commands()

    def cleanup_swapchain(self):
        # raise NotImplementedError()
        self.swapchain.destroy()

    def create_swapchain(self):
        # raise NotImplementedError()
        self.swapchain.create()
