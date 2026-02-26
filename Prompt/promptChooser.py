import torch
from torch import nn
from Prompt.CoOp import PromptMaker
from utils import encode_text_with_prompt_ensemble_equall
REAL_NAME = {'all': 'medical image', 'Brain': 'Brain', 'Liver':'Liver', 'Retina_RESC':'retinal OCT', 'Chest':'Chest X-ray film', 'Retina_OCT2017':'retinal OCT', 'Histopathology':'histopathological image'}

class DummyOptimizer:
    def step(self):
        pass  # Do nothing

    def zero_grad(self):
        pass  # Do nothing

class PromptChooser(nn.Module):
    def __init__(self,
                 clip_model,
                 args,
                 device
                 ):
        super(PromptChooser, self).__init__()
        self.text_mood = args.text_mood
        self.lr =  args.learning_rate
        if self.text_mood == 'fix':
            print('fix text')
            with torch.cuda.amp.autocast(), torch.no_grad():
                self.text_features_fix = encode_text_with_prompt_ensemble_equall(clip_model, REAL_NAME[args.obj], device)
                self.text_optimizer = DummyOptimizer()
        else:
            prompt_abnormal = [
                "{} with focal opacity",
                "{} showing diffuse opacity",
                "{} demonstrating lesion",
                "{} with mass-like opacity",
                "{} with abnormal texture",
                "{} showing irregular contour",
                "{} exhibiting signal abnormality",
                "{} with structural distortion",
                "{} showing pathological change",
                "{} with atypical appearance",
                "{} demonstrating infiltration",
                "{} with signs of disease",
                "{} showing nodular pattern",
                "{} with heterogeneous intensity",
                "{} exhibiting distinct anomaly",
                "{} with observed deformity",
                "{} showing tissue damage",
                "{} with clinical abnormality",
                "{} demonstrating local defect",
                "{} with suspcious region"
            ]
            prompted_state_abnormal = [state.format(REAL_NAME[args.obj]) for state in prompt_abnormal]
            # Instantiate PromptMaker for abnormal prompts
            self.prompt_maker_abnormal = PromptMaker(
                prompts={'abnormal': prompted_state_abnormal},
                clip_model=clip_model,
                n_ctx=16,
                CSC=True,
                class_token_position=["end",  "front", "middle"]
            ).to(device)
            self.prompt_maker_abnormal.train()

            if self.text_mood == 'learnable_all':
                print('learnable all')
                prompt_normal = [
                    "{} with normal signal pattern",
                    "{} showing uniform opacity",
                    "{} with regular structure",
                    "{} without lesion",
                    "{} showing intact anatomy",
                    "{} demonstrating healthy contour",
                    "{} with typical margins",
                    "{} without signal abnormality",
                    "{} showing no pathological change",
                    "{} with typical appearance",
                    "{} demonstrating clear tissue",
                    "{} without signs of disease",
                    "{} showing homogeneous pattern",
                    "{} with uniform intensity",
                    "{} exhibiting no anomaly",
                    "{} with preserved structure",
                    "{} showing healthy tissue",
                    "{} without clinical abnormality",
                    "{} demonstrating normal morphology",
                    "{} with clear observation"
                ]
                # Create dictionaries to store formatted prompts for normal and abnormal cases
                prompted_state_normal = [state.format(REAL_NAME[args.obj]) for state in prompt_normal]

                # Instantiate PromptMaker for normal prompts
                self.prompt_maker_normal = PromptMaker(
                    prompts={'normal': prompted_state_normal},
                    clip_model=clip_model,
                    n_ctx=16,
                    CSC=True,
                    class_token_position=["end",  "front", "middle"]
                ).to(device)
                # Set both models to training mode
                self.prompt_maker_normal.train()

                # Define a single optimizer for both PromptMakers
                self.text_optimizer = torch.optim.Adam(
                    [
                        {'params': self.prompt_maker_normal.prompt_learner.parameters(), 'lr': self.lr},
                        {'params': self.prompt_maker_abnormal.prompt_learner.parameters(), 'lr': self.lr}
                    ],
                    betas=(0.5, 0.999)
                )

            else: # only learnable abnormal
                print('learnable abnormal')
                with torch.cuda.amp.autocast(), torch.no_grad():
                    self.text_features_normal = encode_text_with_prompt_ensemble_equall(clip_model, REAL_NAME[args.obj], device)[:,0].unsqueeze(1)
                    print(self.text_features_normal.shape)
                self.text_optimizer = torch.optim.Adam(
                    [{'params': self.prompt_maker_abnormal.prompt_learner.parameters(), 'lr': self.lr}],
                    betas=(0.5, 0.999)
                    )


    def forward(self):
        if self.text_mood == 'fix':
            return self.text_features_fix
        else:
            text_features_abnormal = self.prompt_maker_abnormal()  # [768,1]
            if self.text_mood == 'learnable_all':
                text_features_normal = self.prompt_maker_normal()  # [768,1]
                return torch.cat([text_features_normal, text_features_abnormal], dim=1)
            else:
                return torch.cat([self.text_features_normal, text_features_abnormal], dim=1)


    def save_prompt(self, save_dict):
        # Add text-related components based on text_mood
        if self.text_mood == 'fix':
            save_dict['text_features_fix'] = self.text_features_fix
        elif self.text_mood == 'learnable_all':
            save_dict['prompt_maker_normal'] = self.prompt_maker_normal.state_dict()
            save_dict['prompt_maker_abnormal'] = self.prompt_maker_abnormal.state_dict()
        else:  # only learnable abnormal
            save_dict['prompt_maker_abnormal'] = self.prompt_maker_abnormal.state_dict()

        return save_dict
