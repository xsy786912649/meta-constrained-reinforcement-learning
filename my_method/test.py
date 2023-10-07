from train import *



if __name__ == "__main__":

    meta_policy_net = torch.load("meta_policy_net.pkl")
    meta_value_net = torch.load("meta_value_net.pkl")


    for task_number in range(args.task_batch_size):
        target_v=setting_reward()
        batch,batch_extra,accumulated_raward_batch=sample_data_for_task_specific(target_v,meta_policy_net,args.batch_size)
        print("task_number: ",task_number)
        print('(before adaptation) \tAverage reward {:.2f}'.format( accumulated_raward_batch))

        task_specific_value_net = Value(num_inputs)
        meta_value_net_copy = Value(num_inputs)
        for i,param in enumerate(task_specific_value_net.parameters()):
            param.data.copy_(list(meta_value_net.parameters())[i].clone().detach().data)
        for i,param in enumerate(meta_value_net_copy.parameters()):
            param.data.copy_(list(meta_value_net.parameters())[i].clone().detach().data)
        task_specific_value_net = update_task_specific_valuenet(task_specific_value_net,meta_value_net_copy,batch,batch_extra,args.batch_size)
        
        previous_value_net = Value(num_inputs)
        for i,param in enumerate(previous_value_net.parameters()):
            param.data.copy_(list(task_specific_value_net.parameters())[i].clone().detach().data)

        previous_policy_net = Policy(num_inputs, num_actions)
        for i,param in enumerate(previous_policy_net.parameters()):
            param.data.copy_(list(meta_policy_net.parameters())[i].clone().detach().data)

        for iteration_number in range(6):
            batch_2,batch_extra_2,accumulated_raward_batch=sample_data_for_task_specific(target_v,previous_policy_net,args.batch_size)
            print("task_number: ",task_number)
            print('(adaptation {}) \tAverage reward {:.2f}'.format(iteration_number, accumulated_raward_batch))
            
            advantages = compute_adavatage(task_specific_value_net,batch_2,batch_extra_2,args.batch_size)
            advantages = advantages - advantages.mean()

            task_specific_policy=Policy(num_inputs, num_actions)
            for i,param in enumerate(task_specific_policy.parameters()):
                param.data.copy_(list(previous_policy_net.parameters())[i].clone().detach().data)
            task_specific_policy=task_specific_adaptation(task_specific_policy,previous_policy_net,batch_2,advantages,index=1)

            task_specific_value_net = update_task_specific_valuenet(task_specific_value_net,previous_value_net,batch_2,batch_extra_2,args.batch_size)
            
            for i,param in enumerate(previous_value_net.parameters()):
                param.data.copy_(list(task_specific_value_net.parameters())[i].clone().detach().data)
            for i,param in enumerate(previous_policy_net.parameters()):
                param.data.copy_(list(task_specific_policy.parameters())[i].clone().detach().data)
            