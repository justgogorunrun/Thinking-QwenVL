def _build_attention_mask_for_compression(
        self, 
        attention_mask, 
        compressed_positions, 
        # vision_tokens_positions,
        new_vision_tokens_positions,  
        seq_length, 
        batch_size,
        device
    ):
        """Build attention mask that supports aggregation tokens"""

        mask = torch.full(
            (batch_size, 1, seq_length, seq_length),
            torch.finfo(torch.float32).min,
            device=device,
        )  # torch.Size([2, 1, 2871, 2871])

        for b in range(batch_size):
            vision_tokens_positions = new_vision_tokens_positions[b] 
            
            batch_vision = [(s, e) for (batch_idx, s, e) in vision_tokens_positions if batch_idx == b] 
            
           
            batch_compressed = {
                level: [(s, e, clip_idx) for (batch_idx, s, e, l, clip_idx) in compressed_positions if batch_idx == b and l == level]
                for level in range(3)
            } 

            # 1. Original visual tokens within the clip are visible to each other
            self.set_self_visibility(mask, b, batch_vision)

            # 2. Visibility of the first-level aggregation tokens
            self.set_self_visibility(mask, b, [(s, e) for (s, e, _) in batch_compressed.get(0, [])]) # Set the visibility within compressed tokens
            for s, e, clip_idx in batch_compressed.get(0, []):
                if clip_idx == 0:
                    
                    vision_before = [(batch_vision[0][0], s)] 
                    self.set_visibility(mask, b, [(s, e)], vision_before)
                else:
                   
                    comp1_before = [(sb, eb) for (sb, eb, cid) in batch_compressed[0] if cid < clip_idx]
                    self.set_visibility(mask, b, [(s, e)], comp1_before)

                   
                    vision_gap = batch_vision[clip_idx:clip_idx+1] 
                    self.set_visibility(mask, b, [(s, e)], vision_gap)
                    

            # 3. Visibility of the second-level aggregation tokens
            self.set_self_visibility(mask, b, [(s, e) for (s, e, _) in batch_compressed.get(1, [])])
            for s, e, clip_idx in batch_compressed.get(1, []):
                level1_tokens = [item for item in batch_compressed.get(0, []) if item[2] == clip_idx]
                if level1_tokens:
                    l1_start = level1_tokens[0][0]
                    self.set_visibility(mask, b, [(s, e)], [(l1_start, s)])
                
                past_comp2 = [(sb, eb) for (sb, eb, cid) in batch_compressed.get(1, []) if cid <= clip_idx]
                self.set_visibility(mask, b, [(s, e)], past_comp2)

            # 4. Visibility of the third-level aggregation tokens
            self.set_self_visibility(mask, b, [(s, e) for (s, e, _) in batch_compressed.get(2, [])])
            for s, e, clip_idx in batch_compressed.get(2, []):
                
                level2_tokens = [item for item in batch_compressed.get(1, []) if item[2] == clip_idx]
                if level2_tokens:
                    l2_start = level2_tokens[0][0]
                    self.set_visibility(mask, b, [(s, e)], [(l2_start, s)])
                
                past_comp3 = [(sb, eb) for (sb, eb, cid) in batch_compressed.get(2, []) if cid <= clip_idx]
                self.set_visibility(mask, b, [(s, e)], past_comp3)

            # 5. Text regions causal +  the latest level visual aggregation tokens
            vision_and_compressed = [(s, e) for (s, e) in batch_vision]
            for level in range(3):
                vision_and_compressed += [(s, e) for (s, e, _) in batch_compressed.get(level, [])]
            vision_and_compressed.sort() 

            text_regions = []
            last_end = 0
            
            text_regions.append((last_end, vision_and_compressed[0][0]))
            text_regions.append((vision_and_compressed[-1][1], seq_length))

            self.set_causal(mask, b, text_regions)

            for level in range(2, -1, -1):
                highest = batch_compressed.get(level, [])
                if highest:
                    self.set_full_visibility(mask, b, [text_regions[-1]], [(s, e) for (s, e, _) in highest])  
                    self.set_visibility(mask, b, [(s, e) for (s, e, _) in highest], [text_regions[0]]) 
                    break 
            

            self.set_visibility(mask, b, batch_vision, [text_regions[0]]) 
            self.set_visibility(mask, b, [text_regions[-1]], [text_regions[0]]) 

 
        mask = self.ensure_causal_mask(mask)

        mask = mask == 0
        return mask
