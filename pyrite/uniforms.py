import vulkan as vk
import numpy as np


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
        
        self.descriptor_set = vk.vkAllocateDescriptorSets(
            self.app._vk_device,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[self.descriptor_set_layout]
            )
        )[0]
        
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=self._vk_buffer,
            offset=0,
            range=self.total_size
        )
        
        write_descriptor_set = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.descriptor_set,
            dstBinding=0,
            dstArrayElement=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            pBufferInfo=buffer_info
        )
        
        vk.vkUpdateDescriptorSets(
            self.app._vk_device,
            descriptorWriteCount=1,
            pDescriptorWrites=[write_descriptor_set],
            descriptorCopyCount=0,
            pDescriptorCopies=None
        )
        return self.descriptor_set

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
