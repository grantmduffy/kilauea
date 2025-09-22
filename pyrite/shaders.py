import vulkan as vk
from pathlib import Path
import tempfile
import subprocess
import numpy as np


class Shader:

    def __init__(self, app, source: str | Path, stage_name: str=None):
        self.app = app
        self.source = source
        self.stage_name = stage_name
        self.code = Path(source).read_text()
        self.spirv = self.compile().tobytes()
        self._vk_module = vk.vkCreateShaderModule(
            self.app._vk_device,
            vk.VkShaderModuleCreateInfo(
                codeSize=len(self.spirv),
                pCode=self.spirv
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
