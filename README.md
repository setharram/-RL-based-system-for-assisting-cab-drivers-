# RL-based-system-for-assisting-cab-drivers
* This project is based on Reinforcement learning using vanila Deep Q-learning (DQN) architecture.
* Objective is train and build model to maximise the profit by cab drivers over long period by choose the rides which are likely to maximise the total profit earned by the driver that day.

# Environment 
* Car can run for 30 days continious, 24*30 hrs. Then it needs to fuel itself. 
* If the cab-driver is completing his trip at that time, he’ll finish that trip and then stop for gas station. So, the terminal state is independent of the number of rides
covered in a month, it is achieved as soon as the cab-driver crosses 24*30 hours.
* There are only 5 locations in the city where the cab can operate.
* Every decisions are made in hourly interval; thus, the decision epochs are discrete.
* The time taken to travel from one place to another is considered in integer hours (only) and
is dependent on the traffic. Also, the traffic is dependent on the hour-of-the-day and the
day-of-the-week.

# Agent
* Every hour, ride requests come from customers in the form of (pick-up, drop) location. Based on the
current ‘state’, he needs to take an action that could maximise his monthly revenue.

# Reward
* Let Cf be the amount of fuel consumed per hour and Rk be the revenue he obtains from customer for every hour of the ride.
* reward function will be (revenue earned from pickup p to drop q) - (Cost of fuel in moving from pickup p to drop q) - (Fuek used in moving from
current point i to pick-up p)
* R(s = Xi Tj Dk) = {Rk ∗ (Time(p, q)) − Cf ∗ (Time(p, q) + Time(i, p))    a = (p, q)}
*                 {                      −C f                 a = (0,0) }
Where Xi represents a driver’s current location, Tj represents time component, Dk represents the day of the week, p represents the pickup and q
represents the drop location.


# Reference:-
* ‘Deep Reinforcement Learning for List-wise Recommendations’ by Xiangyu Zhao, Liang Zhang, Zhuoye Ding
* https://arxiv.org/abs/1801.00209
