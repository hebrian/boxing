# MADDPG and PPO Models

![PPO vs Random Demo](assets/ppo_vs_random_policy.gif)

*Figure 1: Trained PPO agent (white) outperforming a random policy (black)*

## Instructions

### Testing and Training Scripts

- **Test the MADDPG model against the random policy**  
  Run: `maddpg/maddpg_dev_random.ipynb`

- **Train the MADDPG model**  
  Run: `maddpg/maddpg_dev.ipynb`

- **Test and train the PPO model**  
  Run: `self_play_PPO/self_play_PPO.ipynb`

---

## External Libraries

The following external libraries are required to run the code:

```plaintext
gymnasium==1.0.0
keyboard==0.13.5
numpy==2.2.0
torch==2.5.1
