# Lab4 - Homework Template
### 1. About Knowledge Distillation (15%)
- What Modes of Distillation is used in this Lab ?
    
- What role do logits play in knowledge distillation? What effect does a higher temperature parameter have on logits conversion ?
    
- In Feature-Based Knowledge Distillation, from which parts of the Teacher model do we extract features for knowledge transfer?

### 2. Response-Based KD (30%)

Please explain the following:
- How you choose the Temperature and alpha?
- How you design the loss function?

### 3. Feature-based KD (30%)

Please explain the following:
- How you extract features from the choosing intermediate layers?
- How you design the loss function?

### 4. Comparison of student models w/ & w/o KD (5%)

Provide results according to the following structure:
|                            | loss     | accuracy |
| -------------------------- | -------- | -------- |
| Teacher from scratch       | Text     | Text     |
| Student from scratch       | Text     | Text     |
| Response-based student     | Text     | Text     |
| Featured-based student     | Text     | Text     |

### 5. Implementation Observations and Analysis (20%)
Based on the comparison results above:
- Did any KD method perform unexpectedly? 
- What do you think are the reasons? 
- If not, please share your observations during the implementation process, or what difficulties you encountered and how you solved them?