from train import *

def sample_data_for_task_specific_test(target_v,policy_net,batch_size):
    memory = Memory()

    accumulated_raward_batch = 0
    num_episodes = 0
    for i in range(batch_size):
        state = env.reset()[0]
        state = running_state(state)

        reward_sum = 0
        for t in range(args.max_length):
            action = select_action_test(state,policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            reward=-abs(info['x_velocity']-target_v)#-0.5 * 1e-1 * np.sum(np.square(action))
            reward_sum += reward
            next_state = running_state(next_state)
            path_number = i

            memory.push(state, np.array([action]), path_number, next_state, reward)
            if args.render:
                env.render()
            state = next_state
            if done or truncated:
                break

        num_episodes += 1
        accumulated_raward_batch += reward_sum

    accumulated_raward_batch /= num_episodes
    batch = memory.sample()

    return batch,accumulated_raward_batch

"--------------------------------------------------for initialization of running_state------------------------------------------"
with open("running_state_"+model_lower+".pkl",'rb') as file:
    running_state  = pickle.loads(file.read())

if __name__ == "__main__":

    meta_policy_net = torch.load("meta_policy_net_"+model_lower+".pkl")
    meta_lambda_now=args.meta_lambda*333.0/(999.0+333.0) 
    print(model_lower, "running_state: ",running_state.rs.n)

    accumulated_raward_k_adaptation=[[],[],[],[]]
    accumulated_raward_k_adaptation2=[[],[],[],[]]

    for task_number in range(20):
        target_v=task_number*1.0/10
        print("task_number: ",task_number, " target_v: ", target_v)

        previous_policy_net = Policy(num_inputs, num_actions)
        for i,param in enumerate(previous_policy_net.parameters()):
            param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)

        print(torch.exp(previous_policy_net.action_log_std)) 

        for iteration_number in range(4):
            _,accumulated_raward_batch=sample_data_for_task_specific_test(target_v,previous_policy_net,args.batch_size)
            batch,batch_extra,accumulated_raward_batch2=sample_data_for_task_specific(target_v,previous_policy_net,args.batch_size)
            print("task_number: ",task_number)
            print('(adaptation {}) \tAverage reward {:.2f}'.format(iteration_number, accumulated_raward_batch))
            print('(adaptation {}) \tAverage reward {:.2f}'.format(iteration_number, accumulated_raward_batch2))
            if i>0:
                accumulated_raward_k_adaptation[iteration_number].append(accumulated_raward_batch)
                accumulated_raward_k_adaptation2[iteration_number].append(accumulated_raward_batch2)
            
            q_values = compute_adavatage(batch,batch_extra,args.batch_size)
            q_values = (q_values - q_values.mean()) 

            task_specific_policy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(previous_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,previous_policy_net,batch,q_values,meta_lambda_now,index)

            for i,param in enumerate(previous_policy_net.parameters()):
                param.data.copy_(list(task_specific_policy.parameters())[i].clone().detach().data)
    
    a0=np.array(accumulated_raward_k_adaptation[0])
    a1=np.array(accumulated_raward_k_adaptation[1])
    a2=np.array(accumulated_raward_k_adaptation[2])
    a3=np.array(accumulated_raward_k_adaptation[3])
    #print(a0)
    print(a0.mean())
    #print(a1)
    print(a1.mean())
    #print(a2)
    print(a2.mean())
    #print(a3)
    print(a3.mean())
    a0=np.array(accumulated_raward_k_adaptation2[0])
    a1=np.array(accumulated_raward_k_adaptation2[1])
    a2=np.array(accumulated_raward_k_adaptation2[2])
    a3=np.array(accumulated_raward_k_adaptation2[3])
    #print(a0)
    print(a0.mean())
    #print(a1)
    print(a1.mean())
    #print(a2)
    print(a2.mean())
    #print(a3)
    print(a3.mean())