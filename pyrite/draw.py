import vulkan as vk
import numpy as np
from .shaders import Shader
from pathlib import Path
from .buffer import Buffer


class Pass:
    
    def __init__(self, app, clear_color=(0, 0, 0, 0), final_layout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR):
        self.app = app
        self.clear_color = clear_color
        self.final_layout = final_layout
        self.app.swapchain.objects.append(self)

    def create(self):

        # Determine load operation based on clear_color
        load_op = vk.VK_ATTACHMENT_LOAD_OP_LOAD if self.clear_color is None else vk.VK_ATTACHMENT_LOAD_OP_CLEAR
        initial_layout = vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL if self.clear_color is None else vk.VK_IMAGE_LAYOUT_UNDEFINED

        # TODO: dynamically create color attachment
        color_attachment = vk.VkAttachmentDescription(
            format=self.app.surface_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=load_op,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=initial_layout,
            finalLayout=self.final_layout,
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
        # Only create clear value if we're actually clearing
        self._vk_clear_value = vk.VkClearValue(vk.VkClearColorValue(self.clear_color)) if self.clear_color is not None else None

    def destroy(self):
        if hasattr(self, '_vk_render_pass') and self._vk_render_pass:
            vk.vkDestroyRenderPass(self.app._vk_device, self._vk_render_pass, None)
            self._vk_render_pass = None

    def start(self, command_buffer, target_image):
        return PassContext(self, command_buffer, target_image)


class PassContext:

    def __init__(self, render_pass, command_buffer, target_image):
        self.render_pass = render_pass
        self.command_buffer = command_buffer
        self.target_image = target_image

    def __enter__(self):
        # Set clear value count and pointer based on whether we're clearing
        clear_value_count = 1 if self.render_pass._vk_clear_value is not None else 0
        clear_values = self.render_pass._vk_clear_value if self.render_pass._vk_clear_value is not None else None
        
        # Both user images and swapchain images now use scaled resolution
        self.render_extent = vk.VkExtent2D(width=self.target_image.width, height=self.target_image.height)
        
        vk.vkCmdBeginRenderPass(
            self.command_buffer._vk_command_buffer, vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass._vk_render_pass, framebuffer=self.target_image._vk_framebuffer, 
                renderArea=vk.VkRect2D(offset=[0, 0], extent=self.render_extent), 
                clearValueCount=clear_value_count, pClearValues=clear_values
            ),
            getattr(vk, f'VK_SUBPASS_CONTENTS_INLINE')
        )
        
        return self

    def __exit__(self, *args):
        vk.vkCmdEndRenderPass(self.command_buffer._vk_command_buffer)


class  Drawable:

    VERTEX_FORMAT_MAP = {
        'vec2': (vk.VK_FORMAT_R32G32_SFLOAT, 8),
        'vec3': (vk.VK_FORMAT_R32G32B32_SFLOAT, 12),
        'vec4': (vk.VK_FORMAT_R32G32B32A32_SFLOAT, 16),
    }
    
    def __init__(self, app, vertices: np.ndarray, vertex_shader: Shader | str | Path, fragment_shader: Shader | str, 
                 render_pass, indices: np.ndarray=None, targets=None, vertex_attributes=None, uniforms=None, textures=None):
        self.app = app
        self.vertices = Buffer(app, vertices)
        self.indices = Buffer(indices) if indices is not None else None
        self.vertex_shader = Shader(app, vertex_shader, 'vertex') if not isinstance(vertex_shader, Shader) else vertex_shader
        self.fragment_shader = Shader(app, fragment_shader, 'fragment') if not isinstance(fragment_shader, Shader) else fragment_shader
        self.targets = targets
        self.vertex_attributes = vertex_attributes or []
        self.render_pass = render_pass
        self.uniforms = uniforms
        self.textures = textures or []
        self.app.swapchain.objects.append(self)

    def create(self):
        self.create_descriptor_sets()
        self.create_pipeline()

    def create_descriptor_sets(self):
        """Create descriptor set layouts and descriptor sets based on provided resources"""
        self.descriptor_set_layouts = []
        self.descriptor_sets = []
        
        # Create descriptor set for uniforms (binding 0)
        if self.uniforms:
            self.descriptor_set_layouts.append(self.uniforms.descriptor_set_layout)
            self.descriptor_sets.append(self.uniforms.descriptor_set)
        
        # Create descriptor set for textures (binding 1+)
        if self.textures:
            bindings = []
            for i, texture in enumerate(self.textures):
                bindings.append(vk.VkDescriptorSetLayoutBinding(
                    binding=i,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                ))
            
            texture_descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
                self.app._vk_device,
                vk.VkDescriptorSetLayoutCreateInfo(
                    bindingCount=len(bindings),
                    pBindings=bindings
                ),
                None
            )
            self.descriptor_set_layouts.append(texture_descriptor_set_layout)

            texture_descriptor_set = vk.vkAllocateDescriptorSets(
                self.app._vk_device,
                vk.VkDescriptorSetAllocateInfo(
                    descriptorPool=self.app._vk_descriptor_pool,
                    descriptorSetCount=1,
                    pSetLayouts=[texture_descriptor_set_layout,]
                )
            )[0]

            for i, texture in enumerate(self.textures):
                image_info = vk.VkDescriptorImageInfo(
                    sampler=texture._vk_sampler,
                    imageView=texture.image._vk_image_view,
                    imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                )

                write_set = vk.VkWriteDescriptorSet(
                    dstSet=texture_descriptor_set,
                    dstBinding=i,
                    dstArrayElement=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount=1,
                    pImageInfo=image_info
                )
                vk.vkUpdateDescriptorSets(self.app._vk_device, 1, [write_set,], 0, None)
            
            self.descriptor_sets.append(texture_descriptor_set)

    def create_pipeline(self):
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

        # Use dynamic viewport and scissor - they will be set at draw time
        viewport_state_ci = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewportCount=1,
            pViewports=None,  # Will be set dynamically
            scissorCount=1,
            pScissors=None    # Will be set dynamically
        )
        
        # Enable dynamic viewport and scissor
        dynamic_states = [vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]
        dynamic_state_ci = vk.VkPipelineDynamicStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamicStateCount=len(dynamic_states),
            pDynamicStates=dynamic_states
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

        self._vk_pipeline_layout = vk.vkCreatePipelineLayout(
            self.app._vk_device, 
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=len(self.descriptor_set_layouts),
                pSetLayouts=self.descriptor_set_layouts if self.descriptor_set_layouts else None
            ), None
        )

        shader_stages = [self.vertex_shader._vk_stage, self.fragment_shader._vk_stage]
        
        # Create pipeline
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
            pDynamicState=dynamic_state_ci,
            layout=self._vk_pipeline_layout,
            renderPass=self.render_pass._vk_render_pass,
            subpass=0
        )
        
        try:
            pipelines = vk.vkCreateGraphicsPipelines(
                self.app._vk_device, 
                vk.VK_NULL_HANDLE, 
                1, 
                [pipeline_create_info], 
                None
            )
            
            if pipelines and len(pipelines) > 0:
                self._vk_pipeline = pipelines[0]
                
                # Check if the pipeline handle is actually valid (not NULL)
                # The issue was in the detection - let's check if pipeline is actually None or invalid
                if not self._vk_pipeline or self._vk_pipeline == vk.VK_NULL_HANDLE:
                    print(f"\n!!! PIPELINE CREATION RETURNED NULL - RUNNING AUTOMATIC DEBUG !!!")
                    # Use the debug method to find out why
                    debug_pipeline = self.app.debug_graphics_pipeline_creation(
                        shader_stages, vs_inputs, input_assembly_ci,
                        viewport_state_ci, rasterizer_ci, multisampling_ci,
                        color_blend_ci, self.render_pass._vk_render_pass, self._vk_pipeline_layout
                    )
                    raise RuntimeError("Pipeline creation returned NULL handle - see debug output above for details")
                else:
                    # Pipeline creation succeeded
                    print(f"Pipeline created successfully: {self._vk_pipeline}")
            else:
                print(f"\n!!! NO PIPELINES RETURNED - RUNNING AUTOMATIC DEBUG !!!")
                # Use the debug method to find out why
                debug_pipeline = self.app.debug_graphics_pipeline_creation(
                    shader_stages, vs_inputs, input_assembly_ci,
                    viewport_state_ci, rasterizer_ci, multisampling_ci,
                    color_blend_ci, self.render_pass._vk_render_pass, self._vk_pipeline_layout
                )
                raise RuntimeError("No pipelines returned from vkCreateGraphicsPipelines - see debug output above for details")
                
        except Exception as e:
            # If it's not one of our expected errors, also run debug
            if "Pipeline creation returned NULL" not in str(e) and "No pipelines returned" not in str(e):
                print(f"\n!!! PIPELINE CREATION EXCEPTION - RUNNING AUTOMATIC DEBUG !!!")
                print(f"Original exception: {e}")
                # Use the debug method to find out why
                try:
                    debug_pipeline = self.app.debug_graphics_pipeline_creation(
                        shader_stages, vs_inputs, input_assembly_ci,
                        viewport_state_ci, rasterizer_ci, multisampling_ci,
                        color_blend_ci, self.render_pass._vk_render_pass, self._vk_pipeline_layout
                    )
                except Exception as debug_e:
                    print(f"Debug method also failed: {debug_e}")
            
            print(f"Pipeline creation failed: {e}")
            raise RuntimeError(f"Failed to create graphics pipeline: {e}")

    def destroy(self):
        if hasattr(self, '_vk_pipeline') and self._vk_pipeline:
            vk.vkDestroyPipeline(self.app._vk_device, self._vk_pipeline, None)
            self._vk_pipeline = None
        if hasattr(self, '_vk_pipeline_layout') and self._vk_pipeline_layout:
            vk.vkDestroyPipelineLayout(self.app._vk_device, self._vk_pipeline_layout, None)
            self._vk_pipeline_layout = None
    
    def draw(self, command_buffer, frame_index=None):
            
        vk.vkCmdBindPipeline(command_buffer._vk_command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._vk_pipeline)
        
        viewport = vk.VkViewport(
            x=0, y=0,
            width=command_buffer.app._vk_extent.width,
            height=command_buffer.app._vk_extent.height,
            minDepth=0.0, maxDepth=1.0
        )
        vk.vkCmdSetViewport(command_buffer._vk_command_buffer, 0, 1, [viewport])

        scissor = vk.VkRect2D(
            offset=vk.VkOffset2D(x=0, y=0),
            extent=command_buffer.app._vk_extent
        )
        vk.vkCmdSetScissor(command_buffer._vk_command_buffer, 0, 1, [scissor])

        # Bind descriptor sets
        for i, dset in enumerate(self.descriptor_sets):
            vk.vkCmdBindDescriptorSets(
                command_buffer._vk_command_buffer,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self._vk_pipeline_layout,
                i,
                1,
                [dset],
                0,
                None
            )
        
        vk.vkCmdBindVertexBuffers(command_buffer._vk_command_buffer, 0, 1, [self.vertices._vk_buffer], [0])
        vk.vkCmdDraw(command_buffer._vk_command_buffer, len(self.vertices.data), 1, 0, 0)


class Compute:

    def __init__(self, app, compute_shader: Shader | str | Path, uniforms=None, storage_images=None):
        self.app = app
        self.compute_shader = Shader(app, compute_shader, 'compute') if not isinstance(compute_shader, Shader) else compute_shader
        self.uniforms = uniforms
        self.storage_images = storage_images or []
        self.app.swapchain.objects.append(self)

    def create(self):
        self.create_descriptor_sets()
        self.create_pipeline()

    def create_descriptor_sets(self):
        """Create descriptor set layouts and descriptor sets based on provided resources"""
        self.descriptor_set_layouts = []
        self.descriptor_sets = []
        
        # Create a single descriptor set with both uniforms and storage images
        if self.uniforms or self.storage_images:
            bindings = []
            
            # Add uniform buffer binding (binding 0)
            if self.uniforms:
                bindings.append(vk.VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                ))
            
            # Add storage image bindings (binding 1+)
            if self.storage_images:
                for i, storage_image in enumerate(self.storage_images):
                    bindings.append(vk.VkDescriptorSetLayoutBinding(
                        binding=1 + i,  # Start at binding 1
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    ))
            
            # Create descriptor set layout
            descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
                self.app._vk_device,
                vk.VkDescriptorSetLayoutCreateInfo(
                    bindingCount=len(bindings),
                    pBindings=bindings
                ),
                None
            )
            self.descriptor_set_layouts.append(descriptor_set_layout)

            # Allocate descriptor set
            descriptor_set = vk.vkAllocateDescriptorSets(
                self.app._vk_device,
                vk.VkDescriptorSetAllocateInfo(
                    descriptorPool=self.app._vk_descriptor_pool,
                    descriptorSetCount=1,
                    pSetLayouts=[descriptor_set_layout,]
                )
            )[0]

            # Update descriptor set with uniform buffer (if present)
            if self.uniforms:
                buffer_info = vk.VkDescriptorBufferInfo(
                    buffer=self.uniforms._vk_buffer,
                    offset=0,
                    range=self.uniforms.total_size
                )

                write_set = vk.VkWriteDescriptorSet(
                    dstSet=descriptor_set,
                    dstBinding=0,
                    dstArrayElement=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    descriptorCount=1,
                    pBufferInfo=buffer_info
                )
                vk.vkUpdateDescriptorSets(self.app._vk_device, 1, [write_set,], 0, None)

            # Update descriptor set with storage images (if present)
            if self.storage_images:
                for i, storage_image in enumerate(self.storage_images):
                    image_info = vk.VkDescriptorImageInfo(
                        sampler=None,
                        imageView=storage_image.image._vk_image_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
                    )

                    write_set = vk.VkWriteDescriptorSet(
                        dstSet=descriptor_set,
                        dstBinding=1 + i,  # Start at binding 1
                        dstArrayElement=0,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        descriptorCount=1,
                        pImageInfo=image_info
                    )
                    vk.vkUpdateDescriptorSets(self.app._vk_device, 1, [write_set,], 0, None)
            
            self.descriptor_sets.append(descriptor_set)

    def create_pipeline(self):
        self._vk_pipeline_layout = vk.vkCreatePipelineLayout(
            self.app._vk_device, 
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=len(self.descriptor_set_layouts),
                pSetLayouts=self.descriptor_set_layouts if self.descriptor_set_layouts else None
            ), None
        )

        shader_stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.compute_shader._vk_module,
            pName='main'
        )
        
        # Create pipeline
        pipeline_create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=shader_stage,
            layout=self._vk_pipeline_layout
        )
        
        self._vk_pipeline = vk.vkCreateComputePipelines(
            self.app._vk_device, 
            vk.VK_NULL_HANDLE, 
            1, 
            [pipeline_create_info], 
            None
        )[0]

    def destroy(self):
        if hasattr(self, '_vk_pipeline') and self._vk_pipeline:
            vk.vkDestroyPipeline(self.app._vk_device, self._vk_pipeline, None)
            self._vk_pipeline = None
        if hasattr(self, '_vk_pipeline_layout') and self._vk_pipeline_layout:
            vk.vkDestroyPipelineLayout(self.app._vk_device, self._vk_pipeline_layout, None)
            self._vk_pipeline_layout = None
    
    def dispatch(self, command_buffer, x, y, z):
        vk.vkCmdBindPipeline(command_buffer._vk_command_buffer, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._vk_pipeline)
        
        # Bind descriptor sets
        for i, dset in enumerate(self.descriptor_sets):
            vk.vkCmdBindDescriptorSets(
                command_buffer._vk_command_buffer,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self._vk_pipeline_layout,
                i,
                1,
                [dset],
                0,
                None
            )
        
        vk.vkCmdDispatch(command_buffer._vk_command_buffer, x, y, z)


class ComputePass:
    
    def __init__(self, app):
        self.app = app

    def start(self, command_buffer, target_image):
        self.command_buffer = command_buffer
        self.target_image = target_image
        return self

    def __enter__(self):
        self.target_image.transition_layout(self.command_buffer, vk.VK_IMAGE_LAYOUT_GENERAL)

    def __exit__(self, *args):
        self.target_image.transition_layout(self.command_buffer, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
