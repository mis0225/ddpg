from dataclasses import dataclass
from Environment import Environment
from Logger import logger

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

#import gym
#from gym import wrappers

import warnings
warnings.filterwarnings('ignore')

from buffer import ReplayBuffer
from models import ActorNetwork, CriticNetwork


MODEL_NAME_HEADER = "WiflyDual_DQN"
N_EPOCHS = 5
N_FRAMES = 40
BATCH_SIZE = 32
MAX_EXPERIENCES = 1000


@dataclass
class Experience:
    state: np.ndarray

    action: float

    reward: float

    next_state: np.ndarray

    #state_next: np.ndarray

    done: bool

class DDPGAgent:

    def __init__(self):

        self.MAX_EXPERIENCES = MAX_EXPERIENCES

        self.MIN_EXPERIENCES = 1

        self.ACTION_SPACE = 1

        self.OBSERVATION_SPACE = 3

        self.UPDATE_PERIOD = 4

        self.START_EPISODES = 1

        self.TAU = 0.02

        self.GAMMA = 0.95

        self.BATCH_SIZE = 32

        # reset current loss
        #self.current_loss = 0.0

        self.actor_network = ActorNetwork(action_space=self.ACTION_SPACE)

        self.target_actor_network = ActorNetwork(action_space=self.ACTION_SPACE)

        self.critic_network = CriticNetwork()

        self.target_critic_network = CriticNetwork()

        self.stdev = 0.2

        self.buffer = ReplayBuffer(max_experiences=self.MAX_EXPERIENCES)

        self.global_steps = 0

        self.batch_size = self.BATCH_SIZE

        self.hiscore = None

        #self._build_networks()




    def update_network(self, batch_size):

        if len(self.buffer) < self.MIN_EXPERIENCES:
            return

        (states, actions, rewards, next_states, dones) = self.buffer.get_minibatch(batch_size)

        next_actions = self.target_actor_network(next_states)

        next_qvalues = self.target_critic_network(next_states, next_actions).numpy().flatten()

        #: Compute taeget values and update CriticNetwork
        target_values = np.vstack(
            [reward + self.GAMMA * next_qvalue if not done else reward
             for reward, done, next_qvalue
             in zip(rewards, dones, next_qvalues)]).astype(np.float32)

        #print(target_values)

        with tf.GradientTape() as tape:
            self.qvalues = self.critic_network(states, actions)
            self.loss = tf.reduce_mean(tf.square(target_values - self.qvalues))
        
        self.lossa = self.loss
        
        #print(self.qvalues)
        #print(self.loss)
        #print(self.lossa)
        #self.lossa = self.loss
        variables = self.critic_network.trainable_variables
        gradients = tape.gradient(self.loss, variables)
        self.critic_network.optimizer.apply_gradients(zip(gradients, variables))

        #: Update ActorNetwork
        with tf.GradientTape() as tape:
            J = -1 * tf.reduce_mean(self.critic_network(states, self.actor_network(states)))

        variables = self.actor_network.trainable_variables
        gradients = tape.gradient(J, variables)
        self.actor_network.optimizer.apply_gradients(zip(gradients, variables))

    def update_target_network(self):

        # soft-target update Actor
        target_actor_weights = self.target_actor_network.get_weights()
        actor_weights = self.actor_network.get_weights()

        assert len(target_actor_weights) == len(actor_weights)

        self.target_actor_network.set_weights(
            (1 - self.TAU) * np.array(target_actor_weights)
            + (self.TAU) * np.array(actor_weights))

        # soft-target update Critic
        target_critic_weights = self.target_critic_network.get_weights()
        critic_weights = self.critic_network.get_weights()

        assert len(target_critic_weights) == len(critic_weights)

        self.target_critic_network.set_weights(
            (1 - self.TAU) * np.array(target_critic_weights)
            + (self.TAU) * np.array(critic_weights))

    def save_model(self):

        self.actor_network.save_weights("checkpoints/actor")

        self.target_actor_network.save_weights("checkpoints/actor_target")

        self.critic_network.save_weights("checkpoints/critic")

        self.target_critic_network.save_weights("checkpoints/critic_target")

    def load_model(self):

        self.actor_network.load_weights("checkpoints/actor")

        self.target_actor_network.load_weights("checkpoints/actor")

        self.critic_network.load_weights("checkpoints/critic")

        self.target_critic_network.load_weights("checkpoints/critic")


    

def main():

    agent = DDPGAgent()
    log = logger()
    actor = ActorNetwork(action_space=1)
    critic = CriticNetwork()
    buffer = agent.buffer
    env = Environment()
    print('Use saved model? y/n')
    ans_yn = input()
    if (ans_yn == 'y'):
        print('Type model no.')
        agent.load_model()
        #agent.load_model(model_path=MODEL_NAME_HEADER + input())
        print('Model load has been done')
    elif(ans_yn == 'n'):
        print('Progam starts without loading a model')
    else:
        print("Type y or n . Quit the program")
        
    print("start")
    for i in range(N_EPOCHS):
        #agent = DDPGAgent()

        #init
        frame = 0
        loss = 0.0
        reward = 0
        rewards = 0
        #Q_max = 0.0
        terminal = True
        env.reset()
        next_state = env.observe_state()
        done = False

        for j in range(N_FRAMES):
            state_current = next_state
            #print(state_current)

            action = actor.sample_action(state_current)
            #value = critic.call(state_current, action)
            env.excute_action(action)
            next_state, ti = env.observe_update_state()
            reward = env.observe_reward(next_state)
            terminal = env.observe_terminal()
            if terminal == False:
                j -= 1
            #agent.store_experience(state_current, action, reward, state_next, terminal)
            #agent.experience_replay()
            #while not done:

            state_ex = np.array(state_current[0], dtype=np.float32)

            action_ex = np.array(action)
            next_state_ex = np.array(next_state[0], dtype=np.float32)
            exp = Experience(state_ex, action_ex, reward, next_state_ex, done)
            #print(exp)

            buffer.add_experience(exp)
            #agent.buffer.add_experience(exp)   
            #print(value)         
            
            agent.update_network(BATCH_SIZE)
            agent.update_target_network()
            print(frame,next_state[0],action, reward)
            rewards += reward
            # for loging
            frame += 1
            loss += agent.lossa
            #print(loss)
            #print(agent.qvalues)
            #Q_max = np.max(agent.qvalues(state_current))
            log.add_log_state_and_action(next_state, env.get_sentparam(), ti)

        print(str(i) + "epoch end")
        log.add_log(["Epoch End"])

        if(i % 1 == 0):
            #agent.create_checkpoint()
            checkpoint_report = "EPOCH: {:03d}/{:03d} | REWARD: {:03f} | LOSS: {:03f}".format(i, N_EPOCHS - 1, rewards/40, loss/40)
            print(checkpoint_report)
            log.add_log([checkpoint_report])

    agent.save_model()
    log.output_log()



if __name__ == "__main__":
    main()