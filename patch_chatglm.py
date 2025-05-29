import os
import shutil
import sys

patch_text = '''
@@ -900,40 +900,75 @@
-    def forward(
-        self,
-        input_ids=None,
-        position_ids=None,
-        attention_mask=None,
-        past_key_values=None,
-        inputs_embeds=None,
-        use_cache=None,
-        output_attentions=None,
-        output_hidden_states=None,
-        return_dict=None,
-        **kwargs,
-    ):
-        if input_ids is None and inputs_embeds is None:
-            raise ValueError("You must specify either input_ids or inputs_embeds")
-
-        if input_ids is not None:
-            batch_size, seq_len = input_ids.shape[:2]
-            device = input_ids.device
-        else:
-            batch_size, seq_len = inputs_embeds.shape[:2]
-            device = inputs_embeds.device
-
-        if position_ids is None:
-            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
-            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
-
-        if attention_mask is None:
-            attention_mask = self.get_masks(input_ids=input_ids, inputs_embeds=inputs_embeds, device=device)
-
-        if inputs_embeds is None:
-            inputs_embeds = self.embedding(input_ids)
-
-        transformer_outputs = self.transformer(
-            input_ids=input_ids,
-            inputs_embeds=inputs_embeds,
-            position_ids=position_ids,
-            attention_mask=attention_mask,
-            past_key_values=past_key_values,
-            use_cache=use_cache,
-            output_attentions=output_attentions,
-            output_hidden_states=output_hidden_states,
-            return_dict=return_dict,
-            **kwargs,
-        )
-
-        return transformer_outputs
+    def forward(
+        self,
+        input_ids=None,
+        position_ids=None,
+        attention_mask=None,
+        past_key_values=None,
+        inputs_embeds=None,
+        use_cache=None,
+        output_attentions=None,
+        output_hidden_states=None,
+        return_dict=None,
+        **kwargs,
+    ):
+        if input_ids is None and inputs_embeds is None:
+            raise ValueError("You must specify either input_ids or inputs_embeds")
+
+        if input_ids is not None:
+            batch_size, seq_len = input_ids.shape[:2]
+            device = input_ids.device
+        else:
+            batch_size, seq_len = inputs_embeds.shape[:2]
+            device = inputs_embeds.device
+
+        if position_ids is None:
+            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
+            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
+
+        if attention_mask is None:
+            attention_mask = self.get_masks(input_ids=input_ids, inputs_embeds=inputs_embeds, device=device)
+
+        if inputs_embeds is None:
+            inputs_embeds = self.embedding(input_ids)
+
+        transformer_outputs = self.transformer(
+            input_ids=input_ids,
+            inputs_embeds=inputs_embeds,
+            position_ids=position_ids,
+            attention_mask=attention_mask,
+            past_key_values=past_key_values,
+            use_cache=use_cache,
+            output_attentions=output_attentions,
+            output_hidden_states=output_hidden_states,
+            return_dict=return_dict,
+            **kwargs,
+        )
+
+        return transformer_outputs
+
+    def get_masks(self, input_ids=None, inputs_embeds=None, device=None):
+        if input_ids is not None:
+            seq_len = input_ids.shape[1]
+        elif inputs_embeds is not None:
+            seq_len = inputs_embeds.shape[1]
+        else:
+            raise ValueError("Either input_ids or inputs_embeds must be provided")
+
+        mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device)
+        return mask
'''

import os
from pathlib import Path

def find_chatglm_modeling_file():
    # Hugging Face cache root - adjust if needed
    cache_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "THUDM" / "chatglm-6b"

    if not cache_root.exists():
        print(f"Cache folder {cache_root} not found!")
        return None

    # Find 'modeling_chatglm.py' in any subdirectory (usually a commit hash folder)
    for subdir in cache_root.iterdir():
        if subdir.is_dir():
            candidate = subdir / "modeling_chatglm.py"
            if candidate.exists():
                return str(candidate)

    print("modeling_chatglm.py not found in cache.")
    return None

import re

def apply_patch(file_path, patch_text):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to match original forward method including decorators (non-greedy)
    forward_pattern = re.compile(r'def forward\(.*?\):.*?(?=def |\Z)', re.DOTALL)

    # Regex to match original get_masks method
    get_masks_pattern = re.compile(r'def get_masks\(.*?\):.*?(?=def |\Z)', re.DOTALL)

    # Extract new forward and get_masks from patch_text manually
    # For example, you can define patch_text as a dict of method_name -> new_code:
    new_forward_code = """<paste your full forward method code here>"""
    new_get_masks_code = """<paste your full get_masks method code here>"""

    # Replace forward method
    content, forward_subs = forward_pattern.subn(new_forward_code, content)
    if forward_subs == 0:
        print("Warning: original forward method not found.")

    # Replace get_masks method
    content, get_masks_subs = get_masks_pattern.subn(new_get_masks_code, content)
    if get_masks_subs == 0:
        print("Warning: original get_masks method not found. Adding new one.")
        # If get_masks not found, append it at the end
        content += "\n\n" + new_get_masks_code

    # Backup original file
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)
    print(f"Backup saved to {backup_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patch applied successfully.")

    return True
import re

def patch_device_usage(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find 'device = input_ids.device' ignoring whitespace
    pattern = re.compile(r'device\s*=\s*input_ids\.device')

    # Replacement line
    replacement = 'device = input_ids.device if input_ids is not None else inputs_embeds.device'

    new_content, count = pattern.subn(replacement, content)

    if count == 0:
        print("No occurrences of 'device = input_ids.device' found.")
    else:
        print(f"Patched {count} occurrences of device usage.")

    # Backup original file
    backup_path = file_path + ".bak"
    import shutil
    shutil.copyfile(file_path, backup_path)
    print(f"Backup saved to {backup_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"File '{file_path}' patched successfully.")

    return True
import re

def find_device_accesses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    device_access_pattern = re.compile(r'(\w+)\.device')

    print(f"Scanning '{file_path}' for '.device' attribute accesses...\n")

    for lineno, line in enumerate(lines, 1):
        matches = device_access_pattern.findall(line)
        if matches:
            for var_name in matches:
                print(f"Line {lineno}: variable '{var_name}' accesses '.device'")
                print(f"    Code: {line.strip()}")
import re
import shutil

def find_unsafe_device_accesses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    device_pattern = re.compile(r'(\w+)\.device')
    safe_pattern = re.compile(r'\b(\w+)\.device\s+if\s+\1\s+is\s+not\s+None\s+else')

    unsafe_lines = []

    for lineno, line in enumerate(lines, 1):
        matches = device_pattern.findall(line)
        if matches:
            for var in matches:
                if not safe_pattern.search(line):
                    unsafe_lines.append((lineno, var, line.strip()))

    if unsafe_lines:
        print("Unsafe `.device` accesses found:")
        for lineno, var, code in unsafe_lines:
            print(f"Line {lineno}: variable '{var}' accesses '.device' unsafely")
            print(f"    Code: {code}")
    else:
        print("No unsafe `.device` accesses found.")

    return unsafe_lines

def patch_query_layer_device(line):
    # Replace unsafe `device = query_layer.device` with safe fallback
    pattern = re.compile(r'device\s*=\s*query_layer\.device')
    if pattern.search(line):
        return pattern.sub('device = query_layer.device if query_layer is not None else self.device', line)
    return line

def add_device_property(content):
    # Add device property inside the model class if missing
    # This is a simple heuristic that inserts property after class definition line
    if 'def device(self):' in content:
        print("Device property already exists; skipping addition.")
        return content

    class_pattern = re.compile(r'(class\s+ChatGLMModel\(.*?\):)')
    match = class_pattern.search(content)
    if not match:
        print("Could not find model class declaration; skipping device property addition.")
        return content

    insert_pos = match.end()

    device_property_code = """

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            import torch
            return torch.device("cpu")
"""

    new_content = content[:insert_pos] + device_property_code + content[insert_pos:]
    print("Added device property to the model class.")
    return new_content

def apply_patches(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Backup original file
    backup_path = file_path + '.bak'
    shutil.copyfile(file_path, backup_path)
    print(f"Backup created at {backup_path}")

    # Patch lines one by one
    new_lines = []
    for line in lines:
        line = patch_query_layer_device(line)
        new_lines.append(line)

    content = ''.join(new_lines)
    content = add_device_property(content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Patching done for {file_path}")
import os
import shutil

PATCHED_FORWARD = '''
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        from transformers import GenerationConfig
        import copy

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        generation_config = getattr(self, "generation_config", None)
        if generation_config is None:
            generation_config = GenerationConfig()

        # Filter None values from kwargs before updating generation_config
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**filtered_kwargs)

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape[:2]
            device = input_ids.device
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **model_kwargs,
        )

        return transformer_outputs
'''

def patch_forward_method(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    import re
    # Regex to find the forward method definition
    pattern = re.compile(r'def forward\(.*?\):.*?(?=^\s*def |\Z)', re.DOTALL | re.MULTILINE)

    if not pattern.search(content):
        print("No forward method found!")
        return False

    new_content = pattern.sub(PATCHED_FORWARD, content)

    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)
    print(f"Backup saved at {backup_path}")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Patched forward method in {file_path}")
    return True



    
def main():
    file_path = find_chatglm_modeling_file()
    if file_path is None:
        print("Could not find modeling_chatglm.py file in transformers_modules folder.")
        return

    print(f"Found modeling_chatglm.py at: {file_path}")
    success = apply_patch(file_path, patch_text)
    if not success:
        print("Failed to apply patch.")
    success2 = patch_device_usage(file_path)
    if not success2:
        print("Failed to apply device patch.")
    find_device_accesses(file_path)
    # TODO: set this path to your actual modeling_chatglm.py
    
    unsafe = find_unsafe_device_accesses(file_path)
    if unsafe:
        print("\nUnsafe device usages found. Proceeding with patch...")
        apply_patches(file_path)
    else:
        print("No unsafe device usages found. No patch needed.")

    patch_forward_method(file_path)


if __name__ == "__main__":
    main()
