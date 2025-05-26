/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { useEffect, useRef, useState, memo } from "react";
import { useLiveAPIContext } from "../../contexts/LiveAPIContext";
import {
  FunctionDeclaration,
  LiveServerToolCall,
  Modality,
  Type,
} from "@google/genai";

const declaration: FunctionDeclaration = {
  name: "update_code",
  description: "Updates and renders a GLSL shader code in the style of Shadertoy. The response will include a base64 PNG screenshot of the first frame if successful.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      code: {
        type: Type.STRING,
        description:
          "GLSL fragment shader code to render. Should be a complete fragment shader.",
      },
    },
    required: ["code"],
  },
  response: {
    type: Type.OBJECT,
    properties: {
      success: { type: Type.BOOLEAN },
      error: { type: Type.STRING },
    },
    required: ["success"],
  },
};

// Default shader code (Shadertoy-style template)
const shaderHeader = `
precision mediump float;
uniform float iTime;
uniform vec2 iResolution;
uniform float iAspect;

// helpers
float dot2(vec2 v) { return dot(v,v); }

// 2d sdf helpers from https://iquilezles.org/articles/distfunctions2d/
// circle
float sdCircle(vec2 p, float r) {
  return length(p) - r;
}

// rounded box with adjustable corner radii
float sdRoundedBox(vec2 p, vec2 b, vec4 r) {
  r.xy = (p.x > 0.0) ? r.xy : r.zw;
  r.x = (p.y > 0.0) ? r.x : r.y;
  vec2 q = abs(p) - b + r.x;
  return min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - r.x;
}

float sdBox(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// triangle
float sdTriangle(vec2 p, vec2 p0, vec2 p1, vec2 p2) {
  vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
  vec2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
  vec2 pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  vec2 pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  vec2 pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
  float s = sign(e0.x * e2.y - e0.y * e2.x);
  vec2 d = min(min(vec2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                   vec2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
                   vec2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));
  return -sqrt(d.x) * sign(d.y);
}

// Uneven Capsule - exact
float sdUnevenCapsule(vec2 p, float r1, float r2, float h) {
  p.x = abs(p.x);
  float b = (r1 - r2) / h;
  float a = sqrt(1.0 - b * b);
  float k = dot(p, vec2(-b, a));
  if (k < 0.0) return length(p) - r1;
  if (k > a * h) return length(p - vec2(0.0, h)) - r2;
  return dot(p, vec2(a, b)) - r1;
}

// Regular hexagon, exact
float sdHexagon(vec2 p, float r) {
  const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
  p = abs(p);
  p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
  p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
  return length(p) * sign(p.y);
}

// Ellipse, exact
float sdEllipse(vec2 p, vec2 ab) {
  p = abs(p);
  if (p.x > p.y) {
    p = p.yx;
    ab = ab.yx;
  }
  float l = ab.y * ab.y - ab.x * ab.x;
  float m = ab.x * p.x / l;
  float m2 = m * m;
  float n = ab.y * p.y / l;
  float n2 = n * n;
  float c = (m2 + n2 - 1.0) / 3.0;
  float c3 = c * c * c;
  float q = c3 + m2 * n2 * 2.0;
  float d = c3 + m2 * n2;
  float g = m + m * n2;
  float co;
  if (d < 0.0) {
    float h = acos(q / c3) / 3.0;
    float s = cos(h);
    float t = sin(h) * sqrt(3.0);
    float rx = sqrt(-c * (s + t + 2.0) + m2);
    float ry = sqrt(-c * (s - t + 2.0) + m2);
    co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0;
  } else {
    float h = 2.0 * m * n * sqrt(d);
    float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
    float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
    float rx = -s - u - c * 4.0 + 2.0 * m2;
    float ry = (s - u) * sqrt(3.0);
    float rm = sqrt(rx * rx + ry * ry);
    co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) / 2.0;
  }
  vec2 r = ab * vec2(co, sqrt(1.0 - co * co));
  return length(r - p) * sign(p.y - r.y);
}

// quadratic beizer curve
float sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C) {    
    vec2 a = B - A;
    vec2 b = A - 2.0 * B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;
    float kk = 1.0 / dot(b, b);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);      
    float res = 0.0;
    float p = ky - kx * kx;
    float p3 = p * p * p;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q * q + 4.0 * p3;
    if (h >= 0.0) { 
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) / 2.0;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = dot2(d + (c + b * t) * t);
    } else {
        float z = sqrt(-p);
        float v = acos(q / (p * z * 2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        vec3 t = clamp(vec3(m + m, -n - m, n - m) * z - kx, 0.0, 1.0);
        res = min(dot2(d + (c + b * t.x) * t.x),
                 dot2(d + (c + b * t.y) * t.y));
        // The third root cannot be the closest
        // res = min(res, dot2(d + (c + b * t.z) * t.z));
    }
    return sqrt(res);
}

`;

const shaderFooter = `
void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
    gl_FragColor.a = 1.0;
}
`

const defaultShaderCode = `
void mainImage(out vec4 color, vec2 xy)
{
    // Normalize pixel coordinates (from 0 to 1)
    vec2 uv = xy/iResolution.xy;
    
    // Convert to centered coordinates (-1 to 1) and correct for aspect ratio
    vec2 p = (uv * 2.0 - 1.0) * vec2(iResolution.x/iResolution.y, 1.0);
    
    // Draw a circle using SDF
    float d = length(p) - 0.5;
    
    // Anti-alias the circle edge based on pixel size
    float pixelSize = 1.5/min(iResolution.x, iResolution.y);
    float aa = smoothstep(-pixelSize, pixelSize, d);
    
    // Create the circle (black) on white background with anti-aliasing
    vec3 col = vec3(aa);
    
    color.rgb = col;
}
`

const instructions = `
<conversation_format>
You are a helpful GLSL artist.
Conversation format:
- user asks for a specific shader or visual effect
- you ask any clarifying questions you need to understand the request
- now you will say something like "I'm adding rainbow" or "great!" (be brief but let the user know you're working on it)
- make an initial attempt at the shader code using the "update_code" tool call.
- the system will update the display of the shader ie {"success": True}, or inform you of any errors
- you will be provided with a rendering of the current shader code if successful
- you reflect on it and tell the user how well it matches the intent of the request
  - if there is an error in the source code say "oops, I'm fixing an error", "one more" if there is a second error etc. do not be verbose, then make another attempt
  - if the shader doesn't match the intent of the request for any reason, quickly take note (use just a couple words) to the user and make at most one attempt to fix it, before asking the user for clarification
- now check in with the user to see if they are happy with the result
</conversation_format>

<conversation_guidelines>
- AVOID mentioning techical details like anti-aliasing unless asked
- USUALLY follow the user's request without adding extraneous details like animations
</conversation_guidelines>

<shader_info>
The shader is normal GLSL fragment shader code written for WebGL 1.0 in the style of Shadertoy.

The shader should use uniforms like iTime for animation and iResolution for screen resolution. Don't ask for additional information, just make your best judgement and create beautiful, creative shaders.
</shader_info>

<initial_code>
${defaultShaderCode}
</initial_code>

Here's how the code you provide will be interpreted:

<shader_template>
${shaderHeader}
<shader_code_inserted_here>
${shaderFooter}
</shader_template>

<shader_code_guidelines>
1. Always consider the aspect ratio of the screen when writing code
2. When adding anti-aliasing make sure to consider the resolution of the screen and make it 1px wide
3. Use SDFs when appropriate
</shader_code_guidelines>

<shader_2d_sdf_reference>
## Making shapes rounded:
All the shapes above can be converted into rounded shapes by subtracting a constant from their distance function. That, effectively moves the isosurface (isoperimeter I guess) from the level zero to one of the outer rings, which naturally are rounded, as it can be seen in the yellow areas in all the images above. So, basically, for any shape defined by d(x,y) = sdf(x,y), one can make it sounded by computing d(x,y) = sdf(x,y) - r.
float opRound( in vec2 p, in float r )
{
  return sdShape(p) - r;
}
## Making shapes annular:

Similarly, shapes can be made annular (like a ring or the layers of an onion), but taking their absolute value and then subtracting a constant from their field. So, for any shape defined by d(x,y) = sdf(x,y) compute d(x,y) = |sdf(x,y)| - r:
float opOnion(vec2 p, float r){
  return abs(sdShape(p)) - r;
}

</shader_2d_sdf_reference>


<tool_usage>
Output GLSL fragment shader code in the style of Shadertoy.
The "update_code" tool call is used to update the shader code.
Remember, if you're using multi-line strings in python, you need to use triple quotes:

<wrong_example>
<input>
print(default_api.update_code(code='void mainImage(out vec4 o, vec2 i)
{
    // Normalize pixel coordinates (from 0 to 1)
    vec2 uv = i/iResolution.xy;
    // ...
}'))
</input>
<error>
SyntaxError: unterminated string literal (detected at line 56)
</error>
</wrong_example>

<correct_example>
<input>
print(default_api.update_code(code='''void mainImage(out vec4 o, vec2 i)
{
    // Normalize pixel coordinates (from 0 to 1)
    // ...
}'''))
</input>
<output>
{"success": True}
</output>

Thanks! And let's get started!
</tool_usage>
`

function GLSLShaderComponent() {
  const [shaderCode, setShaderCode] = useState<string>(defaultShaderCode);
  const { client, setConfig, setModel } = useLiveAPIContext();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const programRef = useRef<WebGLProgram | null>(null);
  const animationRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(Date.now());
  const clientRef = useRef(client);

  // For tracking the specific function call and if a frame capture is pending
  const lastFunctionCallIdRef = useRef<string | null>(null);
  const captureNextFrameRef = useRef<boolean>(false);

  // Update clientRef when client changes
  useEffect(() => {
    clientRef.current = client;
  }, [client]);

  useEffect(() => {
    setModel("models/gemini-2.5-flash-preview-native-audio-dialog");
    // setModel("models/gemini-2.0-flash-exp");
    setConfig({
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: "Aoede" } },
      },
      systemInstruction: {
        parts: [
          {
            text: instructions,
          },
        ],
      },
      tools: [
        { functionDeclarations: [declaration] },
      ],
    });
  }, [setConfig, setModel]);

  useEffect(() => {
    const onToolCall = (toolCall: LiveServerToolCall) => {
      if (!toolCall.functionCalls || !clientRef.current) {
        return;
      }
      const fc = toolCall.functionCalls.find(
        (fc) => fc.name === declaration.name
      );

      if (fc) {
        if (lastFunctionCallIdRef.current) {
          // A shader update is already in progress. Reject this new call.
          clientRef.current.sendToolResponse({
            functionResponses: [{
              response: { success: false, error: "Shader update already in progress. Please wait." },
              id: fc.id,
              name: declaration.name,
            }],
          });
          return;
        }
        lastFunctionCallIdRef.current = fc.id ?? null;
        captureNextFrameRef.current = true; // Signal that the next successful shader update should capture
        setShaderCode((fc.args as any).code as string);
        // NO immediate response here, it's deferred to after capture or on error
      }
    };

    clientRef.current?.on("toolcall", onToolCall);
    return () => {
      clientRef.current?.off("toolcall", onToolCall);
    };
  }, [clientRef]); // Dependencies should be stable or correctly handled if they change

  // Initialize WebGL
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const gl = canvas.getContext("webgl");
    if (!gl) {
      console.error("WebGL not supported");
      return;
    }

    glRef.current = gl;

    // Set canvas size with device pixel ratio for retina displays
    const resizeCanvas = () => {
      const displayWidth = canvas.clientWidth;
      const displayHeight = canvas.clientHeight;
      const devicePixelRatio = window.devicePixelRatio || 1;
      
      const actualWidth = Math.floor(displayWidth * devicePixelRatio);
      const actualHeight = Math.floor(displayHeight * devicePixelRatio);
      
      if (canvas.width !== actualWidth || canvas.height !== actualHeight) {
        canvas.width = actualWidth;
        canvas.height = actualHeight;
        gl.viewport(0, 0, actualWidth, actualHeight);
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Create and compile shader
  const createShader = (gl: WebGLRenderingContext, type: number, source: string): WebGLShader | null => {
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const errorLog = gl.getShaderInfoLog(shader);
      console.error('Shader compilation error:', errorLog);
      if (clientRef.current && lastFunctionCallIdRef.current) {
        clientRef.current.sendToolResponse({
          functionResponses: [{
            response: { 
              success: false,
              error: `Shader compilation error: ${errorLog}` 
            },
            id: lastFunctionCallIdRef.current,
            name: "update_code",
          }],
        });
        lastFunctionCallIdRef.current = null; // Mark as processed
        captureNextFrameRef.current = false; // Abort capture
      }
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  };

  // Create shader program
  const createProgram = (gl: WebGLRenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram | null => {
    const program = gl.createProgram();
    if (!program) return null;

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const errorLog = gl.getProgramInfoLog(program);
      console.error('Program linking error:', errorLog);
      if (clientRef.current && lastFunctionCallIdRef.current) {
        clientRef.current.sendToolResponse({
          functionResponses: [{
            response: { 
              success: false,
              error: `Program linking error: ${errorLog}` 
            },
            id: lastFunctionCallIdRef.current,
            name: "update_code",
          }],
        });
        lastFunctionCallIdRef.current = null; // Mark as processed
        captureNextFrameRef.current = false; // Abort capture
      }
      gl.deleteProgram(program);
      return null;
    }
    return program;
  };

  // Update shader when code changes
  useEffect(() => {
    if (!glRef.current || !canvasRef.current) return;

    const gl = glRef.current;

    const vertexShaderSource = `
      attribute vec2 a_position;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;
    
    const fullFragmentShaderSource = shaderHeader + "\n" + shaderCode + "\n" + shaderFooter;

    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fullFragmentShaderSource);

    if (!vertexShader || !fragmentShader) {
      // createShader already sent the error response and cleared refs
      return;
    }

    const program = createProgram(gl, vertexShader, fragmentShader);
    if (!program) {
      // createProgram already sent the error response and cleared refs
      gl.deleteShader(vertexShader); // Clean up successfully compiled vertex shader
      gl.deleteShader(fragmentShader); // Clean up successfully compiled fragment shader
      return;
    }

    if (programRef.current) {
      gl.deleteProgram(programRef.current);
    }
    programRef.current = program;

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const positionLocation = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    const timeLocation = gl.getUniformLocation(program, 'iTime');
    const resolutionLocation = gl.getUniformLocation(program, 'iResolution');
    const aspectLocation = gl.getUniformLocation(program, 'iAspect');

    const animate = () => {
      if (!glRef.current || !programRef.current || !canvasRef.current || !clientRef.current) {
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        return;
      }

      const currentTime = (Date.now() - startTimeRef.current) / 1000;
      const width = canvasRef.current.width;
      const height = canvasRef.current.height;
      const aspect = width / height;
      
      gl.useProgram(programRef.current);
      
      if (timeLocation) gl.uniform1f(timeLocation, currentTime);
      if (resolutionLocation) gl.uniform2f(resolutionLocation, width, height);
      if (aspectLocation) gl.uniform1f(aspectLocation, aspect);

      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      if (captureNextFrameRef.current && lastFunctionCallIdRef.current) {
        try {
          clientRef.current.sendToolResponse({
            functionResponses: [{
              response: { success: true },
              id: lastFunctionCallIdRef.current,
              name: "update_code",
            }],
          });

          const image = canvasRef.current.toDataURL('image/jpeg', 1.0);
          const data = image.slice(image.indexOf(",") + 1, Infinity);
          clientRef.current.send({text: "Here's a screenshot of the shader you just rendered. Use it to decide on your next actions, which should be either responding to the user or using the 'update_code' tool call again."})
          clientRef.current.sendRealtimeInput([
            { mimeType: "image/jpeg", data },
          ]);
          
        } catch (e: any) {
          console.error("Error capturing canvas:", e);
          clientRef.current.sendToolResponse({
            functionResponses: [{
              response: { success: false, error: `Failed to capture screenshot: ${e.message || e}` },
              id: lastFunctionCallIdRef.current,
              name: "update_code",
            }],
          });
        } finally {
          captureNextFrameRef.current = false;
          lastFunctionCallIdRef.current = null; // Mark as processed
        }
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }
    // Request the first frame, which might trigger capture if the flag is set
    animationRef.current = requestAnimationFrame(animate); 

    // Cleanup shaders when the component unmounts or shaderCode changes leading to new shaders
    return () => {
      gl.deleteShader(vertexShader);
      gl.deleteShader(fragmentShader);
      // Program is cleaned up when a new one is set or on unmount if programRef.current is managed
    };
  }, [shaderCode]); // clientRef is not needed here as it's stable via a ref, setConfig, setModel are for initial setup.

  // Cleanup WebGL program on unmount
  useEffect(() => {
    return () => {
      if (glRef.current && programRef.current) {
        glRef.current.deleteProgram(programRef.current);
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  return (
    <div className="glsl-shader-container" style={{ width: '100%', height: '400px' }}>
      <canvas 
        ref={canvasRef} 
        style={{ 
          width: '100%', 
          height: '100%', 
          display: 'block',
          border: '1px solid #333'
        }} 
      />
      <div 
        style={{ 
          fontSize: '10px', 
          fontFamily: 'monospace', 
          whiteSpace: 'pre-wrap', 
          overflow: 'auto', 
          maxHeight: '100px', 
          padding: '5px', 
          backgroundColor: '#f5f5f5', 
          border: '1px solid #ddd',
          marginTop: '5px'
        }}
      >
        {shaderCode}
      </div>
    </div>
  );
}

export const GLSLShader = memo(GLSLShaderComponent); 