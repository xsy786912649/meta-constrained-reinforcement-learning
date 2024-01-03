from train import *



if __name__ == "__main__":

    meta_policy_net = torch.load("meta_policy_net.pkl")

    "--------------------------------------------------for initialization of running_state------------------------------------------"
    for i in range(args.batch_size):
        state = env.reset()[0]
        state = running_state(state)
        for t in range(args.max_length):
            action = select_action(state,meta_policy_net)
            action = action.data[0].numpy()
            next_state, reward, done, truncated, info = env.step(action)
            next_state = running_state(next_state)

    accumulated_raward_k_adaptation=[[],[],[],[]]

    for task_number in range(20):
        target_v=task_number*1.0/10
        batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
        print("task_number: ",task_number, " target_v: ", target_v)
        print('(before adaptation) \tAverage reward {:.2f}'.format( accumulated_raward_batch))


        previous_policy_net = Policy(num_inputs, num_actions)
        for i,param in enumerate(previous_policy_net.parameters()):
            param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)

        for iteration_number in range(4):
            batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,previous_policy_net,args.batch_size)
            print("task_number: ",task_number)
            print('(adaptation {}) \tAverage reward {:.2f}'.format(iteration_number, accumulated_raward_batch))
            accumulated_raward_k_adaptation[iteration_number].append(accumulated_raward_batch)
            
            q_values = compute_adavatage(batch,batch_extra,args.batch_size)
            q_values = (q_values - q_values.mean()) 

            task_specific_policy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(previous_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,previous_policy_net,batch,q_values,index=1)

            batch_after,batch_extra_after,_=sample_data_for_task_specific(target_v,task_specific_policy,args.batch_size)
            for i,param in enumerate(previous_policy_net.parameters()):
                param.data.copy_(list(task_specific_policy.parameters())[i].clone().detach().data)
    
    a0=np.array(accumulated_raward_k_adaptation[0])
    a1=np.array(accumulated_raward_k_adaptation[1])
    a2=np.array(accumulated_raward_k_adaptation[2])
    a3=np.array(accumulated_raward_k_adaptation[3])
    print(a0)
    print(a0.mean())
    print(a1)
    print(a1.mean())
    print(a2)
    print(a2.mean())
    print(a3)
    print(a3.mean())