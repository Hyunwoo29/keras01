input = 0.5
goal_prediction=0.8 # 애 빼고 다 튜닝된다!
weight = 0.5
lr = 0.1
epoch = 300

for iteration in range(epoch) :
    prediction = input * weight
    error = (prediction - goal_prediction) **2   # mse

    print("Error : " + str(error) + "\tPrediction : " + str(prediction))

    up_y_predict = input * (weight + lr)
    up_error = (goal_prediction - up_y_predict) ** 2    # mse
    down_y_predict = input * (weight -lr)
    down_error = (goal_prediction - down_y_predict) **2

    if(down_error <= up_error) :
        weight = weight - lr
    if(down_error > up_error) :
        weight = weight +lr