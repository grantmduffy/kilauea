from kilauea import App, UniformBuffer, Uniform, Pass, Image, Texture, Swapchain, Drawable
import numpy as np
import glfw
import vulkan as vk
import time


class PerformanceTestApp(App):

    def __init__(self, scale_factor=1.0, title_suffix=""):
        super().__init__(f'Performance Test {title_suffix}', size='fullscreen')
        
        # Calculate scaled dimensions for render targets
        window_width = self._vk_extent.width
        window_height = self._vk_extent.height
        scaled_width = int(window_width / scale_factor)
        scaled_height = int(window_height / scale_factor)
        
        self.scale_factor = scale_factor
        self.scaled_width = scaled_width
        self.scaled_height = scaled_height

        triangle = 0.5 * np.array([
            [-1, -1, 1.0, 0.0, 0.0],
            [1, -1, 0.0, 1.0, 0.0],
            [0.0, 1, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        full_screen = np.array([
            [-1, -1], [1, -1], [1, 1],
            [-1, -1], [1, 1], [-1, 1]
        ], dtype=np.float32)

        # Create uniform buffer
        self.uniforms = UniformBuffer(self)
        self.uniforms['camera'] = Uniform(np.eye(4, dtype=np.float32), glsl_type='mat4')
        self.uniforms['time'] = Uniform(np.array([0.0], dtype=np.float32), glsl_type='float')
        self.uniforms.create()

        # Pass 1: Triangle rendering at scaled resolution
        self.pass1 = Pass(self, final_layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        # Pass 2: Blur at scaled resolution  
        self.pass2 = Pass(self, final_layout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        # Pass 3: Simple upscale to full resolution
        self.pass3 = Pass(self)
        
        # Create images with explicit dimensions
        self.triangle_image = Image(self, render_pass=self.pass1, width=scaled_width, height=scaled_height)
        self.blur_image = Image(self, render_pass=self.pass2, width=scaled_width, height=scaled_height)
        
        # Create textures for sampling
        self.triangle_texture = Texture(self, self.triangle_image)
        self.blur_texture = Texture(self, self.blur_image)
        
        self.triangle_drawable = Drawable(
            self, triangle, 
            './glsl/triangle.vert', './glsl/triangle.frag', 
            self.pass1, 
            vertex_attributes=['vec2', 'vec3'],
            uniforms=self.uniforms
        )
        self.blur_drawable = Drawable(
            self, full_screen,
            './glsl/blur.vert', './glsl/blur.frag',
            self.pass2,
            vertex_attributes=['vec2'],
            uniforms=self.uniforms,
            textures=[self.triangle_texture]
        )
        # Simple upscale pass - copies blurred result to full resolution swapchain
        self.upscale_drawable = Drawable(
            self, full_screen,
            './glsl/blur.vert', './glsl/simple_copy.frag',
            self.pass3,
            vertex_attributes=['vec2'],
            textures=[self.blur_texture]
        )
        
        self.last_time = glfw.get_time()
        self.last_count = 0
        self.fps_interval = 1.0  # Update every second
        self.frame_times = []
        self.start_time = time.time()

    def draw(self, command_buffer, swapchain_image):
        # Pass 1: Triangle rendering at scaled resolution
        with self.pass1.start(command_buffer, self.triangle_image) as pass_ctx:
            viewport = vk.VkViewport(
                x=0, y=0, 
                width=float(pass_ctx.render_extent.width), 
                height=float(pass_ctx.render_extent.height),
                minDepth=0.0, maxDepth=1.0
            )
            scissor = vk.VkRect2D(offset=[0, 0], extent=pass_ctx.render_extent)
            
            vk.vkCmdSetViewport(command_buffer._vk_command_buffer, 0, 1, [viewport])
            vk.vkCmdSetScissor(command_buffer._vk_command_buffer, 0, 1, [scissor])
            
            self.triangle_drawable.draw(command_buffer, 0)

        # Transition triangle image for shader reading
        self.triangle_image.transition_layout(command_buffer, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        # Pass 2: Blur at scaled resolution
        with self.pass2.start(command_buffer, self.blur_image) as pass_ctx:
            viewport = vk.VkViewport(
                x=0, y=0, 
                width=float(pass_ctx.render_extent.width), 
                height=float(pass_ctx.render_extent.height),
                minDepth=0.0, maxDepth=1.0
            )
            scissor = vk.VkRect2D(offset=[0, 0], extent=pass_ctx.render_extent)
            
            vk.vkCmdSetViewport(command_buffer._vk_command_buffer, 0, 1, [viewport])
            vk.vkCmdSetScissor(command_buffer._vk_command_buffer, 0, 1, [scissor])
            
            self.blur_drawable.draw(command_buffer, 0)

        # Transition blur image for shader reading
        self.blur_image.transition_layout(command_buffer, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)

        # Pass 3: Upscale to full resolution swapchain
        with self.pass3.start(command_buffer, swapchain_image) as pass_ctx:
            viewport = vk.VkViewport(
                x=0, y=0, 
                width=float(pass_ctx.render_extent.width), 
                height=float(pass_ctx.render_extent.height),
                minDepth=0.0, maxDepth=1.0
            )
            scissor = vk.VkRect2D(offset=[0, 0], extent=pass_ctx.render_extent)
            
            vk.vkCmdSetViewport(command_buffer._vk_command_buffer, 0, 1, [viewport])
            vk.vkCmdSetScissor(command_buffer._vk_command_buffer, 0, 1, [scissor])
            
            self.upscale_drawable.draw(command_buffer, 0)

    def main_loop(self):
        t = glfw.get_time()
        
        # Update uniforms
        self.uniforms['time'][0] = t
        self.uniforms['camera'][3, 1] = 0.2 * np.sin(t)
        
        # FPS tracking
        if t - self.last_time > self.fps_interval:
            fps = (self.frame_count - self.last_count) / (t - self.last_time)
            self.last_count = self.frame_count
            self.last_time = t
            
            # Show performance info
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            title = f'{self.title} | FPS: {fps:.1f} | Avg: {avg_fps:.1f} | Frames: {self.frame_count}'
            title += f' | Render: {self.scaled_width}x{self.scaled_height}'
            title += f' | Window: {self._vk_extent.width}x{self._vk_extent.height}'
            title += f' | Factor: {self.scale_factor:.1f}x'
            
            glfw.set_window_title(self.window, title)
            
            # Stop after 10 seconds for comparison
            if elapsed > 10:
                glfw.set_window_should_close(self.window, True)


def run_performance_test():
    print("Starting Performance Comparison...")
    print("Each test will run for 10 seconds\n")
    
    results = []
    
    # Test 1: Full resolution (baseline)
    print("Test 1: Full Resolution (1.0x)")
    app1 = PerformanceTestApp(scale_factor=1.0, title_suffix="(Full Res)")
    app1.run()
    results.append({
        'factor': app1.scale_factor,
        'frames': app1.frame_count,
        'time': time.time() - app1.start_time,
        'avg_fps': app1.frame_count / (time.time() - app1.start_time),
        'render_res': f"{app1.scaled_width}x{app1.scaled_height}",
        'window_res': f"{app1._vk_extent.width}x{app1._vk_extent.height}",
        'pixel_reduction': (1.0 - (app1.scaled_width * app1.scaled_height) / (app1._vk_extent.width * app1._vk_extent.height)) * 100
    })
    
    # Test 2: 1.5x upsampling
    print("Test 2: 1.5x Upsampling")
    app2 = PerformanceTestApp(scale_factor=2, title_suffix="(1.5x Upsampling)")
    app2.run()
    results.append({
        'factor': app2.scale_factor,
        'frames': app2.frame_count,
        'time': time.time() - app2.start_time,
        'avg_fps': app2.frame_count / (time.time() - app2.start_time),
        'render_res': f"{app2.scaled_width}x{app2.scaled_height}",
        'window_res': f"{app2._vk_extent.width}x{app2._vk_extent.height}",
        'pixel_reduction': (1.0 - (app2.scaled_width * app2.scaled_height) / (app2._vk_extent.width * app2._vk_extent.height)) * 100
    })
    
    # Test 3: 2.0x upsampling
    print("Test 3: 2.0x Upsampling")
    app3 = PerformanceTestApp(scale_factor=10, title_suffix="(2.0x Upsampling)")
    app3.run()
    results.append({
        'factor': app3.scale_factor,
        'frames': app3.frame_count,
        'time': time.time() - app3.start_time,
        'avg_fps': app3.frame_count / (time.time() - app3.start_time),
        'render_res': f"{app3.scaled_width}x{app3.scaled_height}",
        'window_res': f"{app3._vk_extent.width}x{app3._vk_extent.height}",
        'pixel_reduction': (1.0 - (app3.scaled_width * app3.scaled_height) / (app3._vk_extent.width * app3._vk_extent.height)) * 100
    })
    
    # Print consolidated results
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Factor':<8} {'FPS':<8} {'Frames':<8} {'Render Res':<12} {'Window Res':<12} {'Pixel Reduction':<15}")
    print("-" * 80)
    
    baseline_fps = results[0]['avg_fps']
    for result in results:
        fps_improvement = ((result['avg_fps'] / baseline_fps) - 1) * 100 if baseline_fps > 0 else 0
        print(f"{result['factor']:<8.1f} {result['avg_fps']:<8.1f} {result['frames']:<8} {result['render_res']:<12} {result['window_res']:<12} {result['pixel_reduction']:<15.1f}%")
        if result['factor'] != 1.0:
            print(f"         (+{fps_improvement:+.1f}% vs baseline)")
    
    print("="*80)
    print()


if __name__ == '__main__':
    run_performance_test()
