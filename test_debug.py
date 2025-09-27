#!/usr/bin/env python3
"""
Test script to demonstrate the advanced debugging capabilities including pipeline creation debugging
"""

from kilauea import App
import vulkan as vk
import numpy as np

class DebugTestApp(App):
    def __init__(self):
        super().__init__('Debug Test App')
        
        # Test SPIR-V debugging on the shader files
        print("\n" + "="*80)
        print("TESTING SPIR-V DEBUGGING CAPABILITIES")
        print("="*80)
        
        # Debug vertex shader
        self.debug_shader_spirv('./glsl/triangle.vert', 'vertex')
        
        # Debug fragment shader  
        self.debug_shader_spirv('./glsl/triangle.frag', 'fragment')
        
        # Debug blur shaders if they exist
        self.debug_shader_spirv('./glsl/blur.vert', 'vertex')
        self.debug_shader_spirv('./glsl/blur.frag', 'fragment')
        
        # Test pipeline creation debugging
        print("\n" + "="*80)
        print("TESTING PIPELINE CREATION DEBUGGING")
        print("="*80)
        
        self.test_pipeline_creation_debug()
        
    def test_pipeline_creation_debug(self):
        """Test the pipeline creation debugging with a simple example"""
        try:
            # Create a simple render pass for testing
            color_attachment = vk.VkAttachmentDescription(
                format=self.surface_format,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            )
            
            color_attachment_ref = vk.VkAttachmentReference(
                attachment=0,
                layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            )
            
            subpass = vk.VkSubpassDescription(
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                colorAttachmentCount=1,
                pColorAttachments=[color_attachment_ref]
            )
            
            render_pass = vk.vkCreateRenderPass(
                self._vk_device,
                vk.VkRenderPassCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                    attachmentCount=1,
                    pAttachments=[color_attachment],
                    subpassCount=1,
                    pSubpasses=[subpass]
                ),
                None
            )
            
            # Create a simple pipeline layout
            pipeline_layout = vk.vkCreatePipelineLayout(
                self._vk_device,
                vk.VkPipelineLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    setLayoutCount=0,
                    pushConstantRangeCount=0
                ),
                None
            )
            
            # Create shader modules (this will likely fail and show us why)
            vertex_spirv = self.load_shader_spirv('./glsl/triangle.vert')
            fragment_spirv = self.load_shader_spirv('./glsl/triangle.frag')
            
            if vertex_spirv and fragment_spirv:
                vertex_module = self.debug_shader_module_creation(vertex_spirv, "vertex")
                fragment_module = self.debug_shader_module_creation(fragment_spirv, "fragment")
                
                if vertex_module and fragment_module:
                    # Create shader stages
                    shader_stages = [
                        vk.VkPipelineShaderStageCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                            module=vertex_module,
                            pName=b'main'
                        ),
                        vk.VkPipelineShaderStageCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                            module=fragment_module,
                            pName=b'main'
                        )
                    ]
                    
                    # Create vertex input state (intentionally incomplete to trigger validation errors)
                    vertex_input_state = vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                        vertexBindingDescriptionCount=0,  # This will likely cause issues
                        vertexAttributeDescriptionCount=0  # This will likely cause issues
                    )
                    
                    # Create input assembly state
                    input_assembly_state = vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                        primitiveRestartEnable=vk.VK_FALSE
                    )
                    
                    # Create viewport state
                    viewport_state = vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1,
                        scissorCount=1
                    )
                    
                    # Create rasterization state
                    rasterization_state = vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        depthClampEnable=vk.VK_FALSE,
                        rasterizerDiscardEnable=vk.VK_FALSE,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode=vk.VK_CULL_MODE_BACK_BIT,
                        frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
                        depthBiasEnable=vk.VK_FALSE,
                        lineWidth=1.0
                    )
                    
                    # Create multisample state
                    multisample_state = vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                        sampleShadingEnable=vk.VK_FALSE
                    )
                    
                    # Create color blend state
                    color_blend_attachment = vk.VkPipelineColorBlendAttachmentState(
                        colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT,
                        blendEnable=vk.VK_FALSE
                    )
                    
                    color_blend_state = vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        logicOpEnable=vk.VK_FALSE,
                        attachmentCount=1,
                        pAttachments=[color_blend_attachment]
                    )
                    
                    # Now test the pipeline creation debugging
                    print("\nTesting pipeline creation with intentionally incomplete vertex input...")
                    pipeline = self.debug_graphics_pipeline_creation(
                        shader_stages, vertex_input_state, input_assembly_state,
                        viewport_state, rasterization_state, multisample_state,
                        color_blend_state, render_pass, pipeline_layout
                    )
                    
                    if pipeline:
                        print("Pipeline created successfully (unexpected!)")
                        # Test pipeline executable properties if available
                        self.debug_pipeline_executable_properties(pipeline)
                    else:
                        print("Pipeline creation failed as expected - check validation messages above for details")
                else:
                    print("Shader module creation failed - check debug output above")
            else:
                print("Could not load SPIR-V shaders - check SPIR-V debug output above")
                
        except Exception as e:
            print(f"Pipeline creation test failed with exception: {e}")
            # Show any validation messages that occurred
            messages = self.get_recent_validation_messages(10)
            if messages:
                print("\nRecent validation messages:")
                for i, msg in enumerate(messages):
                    print(f"  {i+1}. {msg['message']}")
    
    def load_shader_spirv(self, shader_path):
        """Load SPIR-V bytecode from a GLSL shader file"""
        try:
            import tempfile
            import subprocess
            
            # Create temporary file for SPIR-V output
            with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as spv_file:
                spv_path = spv_file.name
            
            # Compile GLSL to SPIR-V
            result = subprocess.run(['glslc', shader_path, '-o', spv_path], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"Failed to compile {shader_path}: {result.stderr}")
                return None
            
            # Read SPIR-V bytecode
            with open(spv_path, 'rb') as f:
                spirv_data = f.read()
            
            # Clean up
            import os
            os.unlink(spv_path)
            
            return spirv_data
            
        except Exception as e:
            print(f"Failed to load SPIR-V for {shader_path}: {e}")
            return None
        
    def draw(self, command_buffer, image):
        # Not implementing actual drawing for this test
        pass
        
    def cleanup_swapchain(self):
        pass
        
    def create_swapchain(self):
        pass

if __name__ == '__main__':
    try:
        app = DebugTestApp()
        print("\nDebug test completed successfully!")
    except Exception as e:
        print(f"Debug test failed: {e}")
        import traceback
        traceback.print_exc()
