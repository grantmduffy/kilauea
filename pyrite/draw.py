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

        # TODO: dynamically create color attachment
        color_attachment = vk.VkAttachmentDescription(
            format=self.app.surface_format,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=final_layout,
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

    def destroy(self):
        vk.vkDestroyRenderPass(self.app._vk_device, self._vk_render_pass, None)

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


class Drawable:

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

        self.create_pipeline()

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
        for texture in self.textures:
            descriptor_set_layouts.append(texture.descriptor_set_layout)
        
        self._vk_pipeline_layout = vk.vkCreatePipelineLayout(
            self.app._vk_device, 
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=len(descriptor_set_layouts),
                pSetLayouts=descriptor_set_layouts if descriptor_set_layouts else None
            ), None
        )

        shader_stages = [self.vertex_shader._vk_stage, self.fragment_shader._vk_stage]
        print(f"Creating pipeline with {len(descriptor_set_layouts)} descriptor set layouts")
        print(f"Pipeline layout: {self._vk_pipeline_layout}")
        
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
            layout=self._vk_pipeline_layout,
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
                self._vk_pipeline = pipeline_result[0]
            else:
                self._vk_pipeline = pipeline_result
                
            print(f"Pipeline extracted: {self._vk_pipeline}")
            
            # Simple validation - check if it's NULL
            if str(self._vk_pipeline).find('NULL') != -1:
                print("ERROR: Pipeline creation returned NULL")
                self._vk_pipeline = None
            else:
                print(f"Pipeline created successfully: {self._vk_pipeline}")
                
        except Exception as e:
            print(f"Pipeline creation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            self._vk_pipeline = None

    def destroy(self):
        vk.vkDestroyPipeline(self.app._vk_device, self._vk_pipeline, None)
        vk.vkDestroyPipelineLayout(self.app._vk_device, self._vk_pipeline_layout, None)
    
    def draw(self, command_buffer, frame_index=None):
            
        vk.vkCmdBindPipeline(command_buffer._vk_command_buffer, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._vk_pipeline)
        
        # Bind descriptor sets if uniforms are available
        if self.uniforms and frame_index is not None:
            vk.vkCmdBindDescriptorSets(
                command_buffer._vk_command_buffer,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self._vk_pipeline_layout,
                0,  # firstSet
                1,  # descriptorSetCount
                [self.uniforms.descriptor_set],
                0,  # dynamicOffsetCount
                None  # pDynamicOffsets
            )

        # Bind texture descriptor sets
        if self.textures:
            for i, texture in enumerate(self.textures):
                vk.vkCmdBindDescriptorSets(
                    command_buffer._vk_command_buffer,
                    vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    self._vk_pipeline_layout,
                    1,  # firstSet (set 1 for textures)
                    1,  # descriptorSetCount
                    [texture.descriptor_set],
                    0,  # dynamicOffsetCount
                    None  # pDynamicOffsets
                )
        
        vk.vkCmdBindVertexBuffers(command_buffer._vk_command_buffer, 0, 1, [self.vertices._vk_buffer], [0])
        vk.vkCmdDraw(command_buffer._vk_command_buffer, len(self.vertices.data), 1, 0, 0)
