import libs_dataset.magnetometer_read as Dataset
#import libs_dataset.dataset_testing as Dataset
import models.net_0.model as Model

training_data_files_list = []
training_categories_ids  = []

testing_data_files_list = []
testing_categories_ids  = []


training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat1.csv")
training_categories_ids.append(0)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat2.csv")
training_categories_ids.append(1)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat3.csv")
training_categories_ids.append(2)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat4.csv")
training_categories_ids.append(3)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCatTrailer.csv")
training_categories_ids.append(4)


testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat1.csv")
testing_categories_ids.append(0)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat2.csv")
testing_categories_ids.append(1)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat3.csv")
testing_categories_ids.append(2)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat4.csv")
testing_categories_ids.append(3)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCatTrailer.csv")
testing_categories_ids.append(4)


'''
training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat1.csv")
training_categories_ids.append(0)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat2.csv")
training_categories_ids.append(1)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat3.csv")
training_categories_ids.append(1)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat4.csv")
training_categories_ids.append(2)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCatTrailer.csv")
training_categories_ids.append(0)


testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat1.csv")
testing_categories_ids.append(0)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat2.csv")
testing_categories_ids.append(1)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat3.csv")
testing_categories_ids.append(1)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat4.csv")
testing_categories_ids.append(2)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCatTrailer.csv")
testing_categories_ids.append(0)
'''


'''
training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat1.csv")
training_categories_ids.append(0)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat2.csv")
training_categories_ids.append(0)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat3.csv")
training_categories_ids.append(1)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCat4.csv")
training_categories_ids.append(1)

training_data_files_list.append("/Users/michal/dataset/car_detection/dataS1RawWinCatTrailer.csv")
training_categories_ids.append(1)


testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat1.csv")
testing_categories_ids.append(0)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat2.csv")
testing_categories_ids.append(0)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat3.csv")
testing_categories_ids.append(1)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCat4.csv")
testing_categories_ids.append(1)

testing_data_files_list.append("/Users/michal/dataset/car_detection/dataS2RawWinCatTrailer.csv")
testing_categories_ids.append(1)
'''

dataset = Dataset.Create(training_data_files_list, training_categories_ids, testing_data_files_list, testing_categories_ids)

#dataset = Dataset.Create()

model = Model.Create(dataset.input_shape, dataset.classes_count)

import torch
import libs.confussion_matrix

epoch_count = 100
leraning_rates = [0.0001, 0.0001, 0.0001, 0.00001, 0.00001]

accuracy_best = 0.0

for epoch in range(epoch_count):
    
    batch_size  = 32
    batch_count = (dataset.get_training_count()+batch_size) // batch_size

    learning_rate = leraning_rates[epoch%len(leraning_rates)]
    
    optimizer  = torch.optim.Adam(model.parameters(), lr= learning_rate, weight_decay=10**-5)  

    training_confussion_matrix = libs.confussion_matrix.ConfussionMatrix(dataset.classes_count)
    for batch_id in range(batch_count):
        training_x, training_y = dataset.get_training_batch()

        predicted_y = model.forward(training_x)

        error = (training_y - predicted_y)**2
        loss  = error.mean()

        loss.backward()
        optimizer.step()

        training_confussion_matrix.add_batch(training_y.detach().to("cpu").numpy(), predicted_y.detach().to("cpu").numpy())

    training_confussion_matrix.compute()
    
    batch_count = (dataset.get_testing_count()+batch_size) // batch_size
    testing_confussion_matrix = libs.confussion_matrix.ConfussionMatrix(dataset.classes_count)
    for batch_id in range(batch_count):
        testing_x, testing_y = dataset.get_testing_batch()

        predicted_y = model.forward(testing_x)

        error = (testing_y - predicted_y)**2
        loss  = error.mean()

        testing_confussion_matrix.add_batch(testing_y.detach().to("cpu").numpy(), predicted_y.detach().to("cpu").numpy())
        

    testing_confussion_matrix.compute()

    print("epoch = ", epoch, "\n")

    if testing_confussion_matrix.accuraccy > accuracy_best:
        accuracy_best = testing_confussion_matrix.accuraccy

        print("\n\n\n")
        print("=================================================")
        print("new best net in ", epoch, "\n")
        print("TRAINING result ")
        print(training_confussion_matrix.get_result())
        print("TESTING result ")
        print(testing_confussion_matrix.get_result())
        print("\n\n\n")