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

def find_chatglm_modeling_file():
    import transformers_modules
    base_dir = transformers_modules.__path__[0]
    # The path usually looks like:
    # ~/.cache/huggingface/modules/transformers_modules/THUDM/chatglm-6b/<commit_hash>/modeling_chatglm.py
    for root, dirs, files in os.walk(base_dir):
        if 'modeling_chatglm.py' in files:
            return os.path.join(root, 'modeling_chatglm.py')
    return None

def apply_patch(file_path, patch_text):
    import re
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # This simple patch replaces the forward and get_masks methods
    # by searching the 'def forward' line and replacing until end of method.
    # For more robust patching, use patch tool or difflib.

    # Here we just overwrite the entire forward + get_masks methods block

    pattern = re.compile(
        r'def forward\(.*?\):.*?return transformer_outputs\n', 
        re.DOTALL
    )

    new_forward = patch_text.split('--- modeling_chatglm.py.orig\n')[1].split('+++ modeling_chatglm.py\n')[1]
    # Strip diff markers
    new_forward = '\n'.join([line[1:] if line.startswith('+') else line for line in new_forward.splitlines() if line.strip() and not line.startswith('-')])

    if not pattern.search(content):
        print("Could not find forward method to patch. Exiting.")
        return False

    content = pattern.sub(new_forward, content)

    # Append get_masks method at the end if needed
    if 'def get_masks' not in content:
        content += '\n\n' + new_forward.split('def get_masks')[1]

    backup_path = file_path + '.bak'
    print(f"Backing up original file to {backup_path}")
    shutil.copyfile(file_path, backup_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patch applied successfully to {file_path}")
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

if __name__ == "__main__":
    main()
