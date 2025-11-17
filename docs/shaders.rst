kilauea.shaders
================

.. automodule:: kilauea.shaders

.. autoclass:: Shader
   :members:
   :inherited-members:

   Shader compilation and Vulkan module creation from GLSL source.

   Shader handles the compilation of GLSL shader source code to SPIR-V using the
   glslc compiler, then creates Vulkan shader modules. It provides the pipeline
   stage information needed for graphics pipeline creation.

   **Key Features:**

   * Automatic GLSL to SPIR-V compilation using glslc
   * Vulkan shader module creation
   * Pipeline shader stage information generation
   * Error reporting with compilation warnings

   **Supported Shader Stages:**

   * ``vertex`` - Vertex shader stage
   * ``fragment`` - Fragment shader stage
   * ``compute`` - Compute shader stage

   **Usage Example:**

   .. code-block:: python

      from kilauea.shaders import Shader

      # Compile vertex shader
      vertex_shader = Shader(app, 'triangle.vert', 'vertex')

      # Compile fragment shader
      fragment_shader = Shader(app, 'triangle.frag', 'fragment')

      # Shaders are typically used indirectly through Drawable/Compute
      triangle = Drawable(app, vertices, vertex_shader, fragment_shader, ...)

   **Shader Source Format:**

   Shaders can be loaded from file paths or passed as strings:

   .. code-block:: python

      # From file path
      shader = Shader(app, 'shaders/myshader.vert', 'vertex')

      # From string (Path object)
      shader_path = Path('shaders/myshader.vert')
      shader = Shader(app, shader_path, 'vertex')

   **Requirements:**

   The Vulkan SDK must be installed with ``glslc`` in the system PATH for shader compilation.

   .. automethod:: __init__(app, source: str | Path, stage_name: str=None)

      :param app: The Kilauea application instance
      :type app: App
      :param source: GLSL source file path or content
      :type source: str or Path
      :param stage_name: Shader stage ('vertex', 'fragment', 'compute')
      :type stage_name: str

   .. automethod:: compile()

      Compiles the GLSL source to SPIR-V bytecode using glslc.

      :return: Compiled SPIR-V code as numpy uint32 array
      :rtype: np.ndarray
      :raises: RuntimeError if compilation fails or glslc is not found
