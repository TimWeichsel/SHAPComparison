from evaluation_results_loader import get_evaluation
import matplotlib.pyplot as plt



rbo_runs =[
    ('S', 'LR',  'D', '25'),
    ('S', 'LR',  'D', '50'),
    ('S', 'LR',  'D', '75'),
    ('S', 'LR',  'D', '100'),
    ('S', 'LR',  'D', '200'),
    ('S', 'LR',  'D', '300'),
    ('S', 'LR',  'D', '400'),
    ('S', 'LR',  'D', '500'),
    ('S', 'LR',  'D', '600'),
    ('S', 'LR',  'D', '700'),
    ('S', 'LR',  'D', '800'),
    ('S', 'LR',  'D', '900'),
    ('S', 'LR',  'D', '1000'),
    ('S', 'LR',  'D', '1500'),
    ('S', 'LR',  'D', '2000'),
    ('S', 'LR',  'D', '2500'),
    
    ('S', 'RR',  'D', '30'),
    ('S', 'RR',  'D', '50'),
    ('S', 'RR',  'D', '100'),
    ('S', 'RR',  'D', '200'),
    ('S', 'RR',  'D', '300'),
    ('S', 'RR',  'D', '400'),
    ('S', 'RR',  'D', '500'),
    ('S', 'RR',  'D', '600'),
    ('S', 'RR',  'D', '700'),
    ('S', 'RR',  'D', '800'),
    ('S', 'RR',  'D', '900'),
    ('S', 'RR',  'D', '1000'),
    ('S', 'RR',  'D', '1150'),
    ('S', 'RR',  'D', '1300'),
    ('S', 'RR',  'D', '1500'),
    ('S', 'RR',  'D', '2000'),
    ('S', 'RR',  'D', '2500'),
    ('S', 'RR',  'D', '3000'),
    ('S', 'RR',  'D', '3500'),
    ('S', 'RR',  'D', '4000'),
    ('S', 'RR',  'D', '4500'),
    ('S', 'RR',  'D', '5000'),
    ('S', 'RR',  'D', '5500'),
    ('S', 'RR',  'D', '6000'),
    ('S', 'RR',  'D', '7000'),
    ('S', 'RR',  'D', '8000'),
    ('S', 'RR',  'D', '9000'),
    ('S', 'RR',  'D', '10000'),
    ('S', 'RR',  'D', '12000'),

    
    ('S', 'SR',  'D', '30'),
    ('S', 'SR',  'D', '50'),
    ('S', 'SR',  'D', '100'),
    ('S', 'SR',  'D', '150'),
    ('S', 'SR',  'D', '200'),
    ('S', 'SR',  'D', '250'),
    ('S', 'SR',  'D', '300'),
    ('S', 'SR',  'D', '350'),
    ('S', 'SR',  'D', '400'),
    ('S', 'SR',  'D', '450'),
    ('S', 'SR',  'D', '500'),
    ('S', 'SR',  'D', '550'),
    ('S', 'SR',  'D', '600'),
    ('S', 'SR',  'D', '650'),
    ('S', 'SR',  'D', '700'),
    ('S', 'SR',  'D', '750'),
    ('S', 'SR',  'D', '800'),
    ('S', 'SR',  'D', '850'),
    ('S', 'SR',  'D', '1000'),
    ('S', 'SR',  'D', '3000'),
    ('S', 'SR',  'D', '5000'),
    ('S', 'SR',  'D', '10000'),
    ('S', 'SR',  'D', '15000'),
    ('S', 'SR',  'D', '20000'),
    ('S', 'SR',  'D', '25000'),
    ('S', 'SR',  'D', '30000'),
    ('S', 'SR',  'D', '35000'),
    ('S', 'SR',  'D', '40000'),
    ('S', 'SR',  'D', '45000'),
    ('S', 'SR',  'D', '50000'),

    
    ('S', 'XR',  'D', '100'),
    ('S', 'XR',  'D', '500'),
    ('S', 'XR',  'D', '1000'),
    ('S', 'XR',  'D', '2000'),
    ('S', 'XR',  'D', '3000'),
    ('S', 'XR',  'D', '4000'),
    ('S', 'XR',  'D', '5000'),
    ('S', 'XR',  'D', '6000'),
    ('S', 'XR',  'D', '7000'),
    ('S', 'XR',  'D', '8000'),
    ('S', 'XR',  'D', '9000'),
    ('S', 'XR',  'D', '10000'),
    ('S', 'XR',  'D', '11000'),
    ('S', 'XR',  'D', '12000'),
    ('S', 'XR',  'D', '13000'),
    ('S', 'XR',  'D', '14000'),
    ('S', 'XR',  'D', '15000'),
    ('S', 'XR',  'D', '16000'),
    ('S', 'XR',  'D', '17000'),
    ('S', 'XR',  'D', '18000'),
    
    ('S', 'LR',  'W', '30'),
    ('S', 'LR',  'W', '50'),
    ('S', 'LR',  'W', '100'),
    ('S', 'LR',  'W', '200'),
    ('S', 'LR',  'W', '300'),
    ('S', 'LR',  'W', '400'),
    ('S', 'LR',  'W', '500'),
    ('S', 'LR',  'W', '600'),
    ('S', 'LR',  'W', '700'),
    ('S', 'LR',  'W', '800'),
    ('S', 'LR',  'W', '900'),
    ('S', 'LR',  'W', '1000'),
    ('S', 'LR',  'W', '1100'),

    
    ('S', 'RR',  'W', '30'),
    ('S', 'RR',  'W', '50'),
    ('S', 'RR',  'W', '100'),
    ('S', 'RR',  'W', '200'),
    ('S', 'RR',  'W', '300'),
    ('S', 'RR',  'W', '400'),
    ('S', 'RR',  'W', '500'),
    ('S', 'RR',  'W', '600'),
    ('S', 'RR',  'W', '700'),
    ('S', 'RR',  'W', '800'),
    ('S', 'RR',  'W', '900'),
    ('S', 'RR',  'W', '1000'),
    ('S', 'RR',  'W', '1500'),
    ('S', 'RR',  'W', '2000'),
    ('S', 'RR',  'W', '2500'),
    ('S', 'RR',  'W', '3000'),
    ('S', 'RR',  'W', '3500'),
    ('S', 'RR',  'W', '4000'),
    ('S', 'RR',  'W', '4500'),
    ('S', 'RR',  'W', '5000'),
    ('S', 'RR',  'W', '5500'),
    ('S', 'RR',  'W', '6000'),
    ('S', 'RR',  'W', '9000'),
    ('S', 'RR',  'W', '12000'),
    ('S', 'RR',  'W', '15000'),
    ('S', 'RR',  'W', '18000'),
    ('S', 'RR',  'W', '21000'),
    ('S', 'RR',  'W', '24000'),
    ('S', 'RR',  'W', '27000'),
    ('S', 'RR',  'W', '30000'),
    ('S', 'RR',  'W', '33000'),
    ('S', 'RR',  'W', '36000'),
    
    ('S', 'XR',  'W', '30'),
    ('S', 'XR',  'W', '250'),
    ('S', 'XR',  'W', '600'),
    ('S', 'XR',  'W', '1000'),
    ('S', 'XR',  'W', '1500'),
    ('S', 'XR',  'W', '2000'),
    ('S', 'XR',  'W', '3000'),
    ('S', 'XR',  'W', '4000'),
    ('S', 'XR',  'W', '6000'),
    ('S', 'XR',  'W', '8000'),
    ('S', 'XR',  'W', '10000'),
    ('S', 'XR',  'W', '12000'),
    ('S', 'XR',  'W', '16000'),
    ('S', 'XR',  'W', '20000'),
    ('S', 'XR',  'W', '27850'), 
    
    ('S', 'SR',  'W', '50'),
    ('S', 'SR',  'W', '100'),
    ('S', 'SR',  'W', '200'),
    ('S', 'SR',  'W', '300'),
    ('S', 'SR',  'W', '400'),
    ('S', 'SR',  'W', '500'),
    ('S', 'SR',  'W', '750'),
    ('S', 'SR',  'W', '1000'),
    ('S', 'SR',  'W', '2000'),
    ('S', 'SR',  'W', '3000'),
    ('S', 'SR',  'W', '4000'),
    
    #K
    #1,100,500,1000,5000,10000,20000,50000,100000,150000,200000,240000
    ('K', 'LR',  'D', '1'),
    ('K', 'LR',  'D', '100'),
    ('K', 'LR',  'D', '500'),
    ('K', 'LR',  'D', '1000'),
    ('K', 'LR',  'D', '5000'),
    ('K', 'LR',  'D', '10000'),
    ('K', 'LR',  'D', '20000'),
    ('K', 'LR',  'D', '50000'),
    ('K', 'LR',  'D', '100000'),
    ('K', 'LR',  'D', '150000'),
    ('K', 'LR',  'D', '200000'),
    ('K', 'LR',  'D', '240000'),
    
    #1,10,20,30,40,50,75,100,125,150,200,250,300
    ('K', 'RR',  'D', '1'),
    ('K', 'RR',  'D', '10'),
    ('K', 'RR',  'D', '20'),
    ('K', 'RR',  'D', '30'),
    ('K', 'RR',  'D', '40'),
    ('K', 'RR',  'D', '50'),
    ('K', 'RR',  'D', '75'),
    ('K', 'RR',  'D', '100'),
    ('K', 'RR',  'D', '125'),
    ('K', 'RR',  'D', '150'),
    ('K', 'RR',  'D', '200'),
    ('K', 'RR',  'D', '250'),
    ('K', 'RR',  'D', '300'),
    
    #1,10,20,30,40,50,75,100,150,200,250,300,350,380
    ('K', 'SR',  'D', '1'),
    ('K', 'SR',  'D', '10'),
    ('K', 'SR',  'D', '20'),
    ('K', 'SR',  'D', '30'),
    ('K', 'SR',  'D', '40'),
    ('K', 'SR',  'D', '50'),
    ('K', 'SR',  'D', '75'),
    ('K', 'SR',  'D', '100'),
    ('K', 'SR',  'D', '150'),
    ('K', 'SR',  'D', '200'),
    ('K', 'SR',  'D', '250'),
    ('K', 'SR',  'D', '300'),
    ('K', 'SR',  'D', '350'),
    
    #1,10,25,50,75,100,150,200,250,300,350,400,450
    ('K', 'XR',  'D', '1'),
    ('K', 'XR',  'D', '10'),
    ('K', 'XR',  'D', '25'),
    ('K', 'XR',  'D', '50'),
    ('K', 'XR',  'D', '75'),
    ('K', 'XR',  'D', '100'),
    ('K', 'XR',  'D', '150'),
    ('K', 'XR',  'D', '200'),
    ('K', 'XR',  'D', '250'),
    ('K', 'XR',  'D', '300'),
    ('K', 'XR',  'D', '350'),
    ('K', 'XR',  'D', '400'),
    ('K', 'XR',  'D', '450'),
    
    #1,3,5,10,15,20,30,40,50,60,70
    ('K', 'LR',  'W', '1'),
    ('K', 'LR',  'W', '3'),
    ('K', 'LR',  'W', '5'),
    ('K', 'LR',  'W', '10'),
    ('K', 'LR',  'W', '15'),
    ('K', 'LR',  'W', '20'),
    ('K', 'LR',  'W', '30'),
    ('K', 'LR',  'W', '40'),
    ('K', 'LR',  'W', '50'),
    ('K', 'LR',  'W', '60'),
    ('K', 'LR',  'W', '70'),
    
    #1,2,3,4,5,7,10,15,25,39
    ('K', 'RR',  'W', '1'),
    ('K', 'RR',  'W', '2'),
    ('K', 'RR',  'W', '3'),
    
    ('K', 'RR',  'W', '4'),
    ('K', 'RR',  'W', '5'),
    ('K', 'RR',  'W', '7'),
    ('K', 'RR',  'W', '10'),
    ('K', 'RR',  'W', '15'),
    ('K', 'RR',  'W', '25'),
    ('K', 'RR',  'W', '39'),
    
    ('K', 'SR',  'W', '1'),
    ('K', 'SR',  'W', '2'),
    ('K', 'SR',  'W', '10'),
    
    
    #1,3,5,7,10,16,20,26,30,35,40,43
    ('K', 'XR',  'W', '1'),
    ('K', 'XR',  'W', '3'),
    ('K', 'XR',  'W', '5'),
    ('K', 'XR',  'W', '7'),
    ('K', 'XR',  'W', '10'),
    ('K', 'XR',  'W', '15'),
    ('K', 'XR',  'W', '20'),
    ('K', 'XR',  'W', '25'),
    ('K', 'XR',  'W', '30'),
    ('K', 'XR',  'W', '35'),
    ('K', 'XR',  'W', '40'),


    ('C', 'LR', 'D', '12'),
    ('C', 'LR', 'D', '16'), #ERROR
    ('C', 'LR', 'D', '20'),
    ('C', 'LR', 'D', '26'), #ERROR
    ('C', 'LR', 'D', '30'),
    ('C', 'LR', 'D', '40'),
    ('C', 'LR', 'D', '50'),
    ('C', 'LR', 'D', '60'),
    ('C', 'LR', 'D', '70'),
    ('C', 'LR', 'D', '80'),
    ('C', 'LR', 'D', '90'),
    ('C', 'LR', 'D', '100'),
    ('C', 'LR', 'D', '120'),
    ('C', 'LR', 'D', '140'),
    ('C', 'LR', 'D', '160'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048 
    ('C', 'RR', 'D', '12'),
    ('C', 'RR', 'D', '50'),
    ('C', 'RR', 'D', '100'),
    ('C', 'RR', 'D', '200'),
    ('C', 'RR', 'D', '300'),
    ('C', 'RR', 'D', '400'),
    ('C', 'RR', 'D', '500'),
    ('C', 'RR', 'D', '750'),
    ('C', 'RR', 'D', '1000'),
    ('C', 'RR', 'D', '1250'),
    ('C', 'RR', 'D', '1500'),
    ('C', 'RR', 'D', '1750'),
    ('C', 'RR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'SR', 'D', '12'),
    ('C', 'SR', 'D', '50'),
    ('C', 'SR', 'D', '100'),
    ('C', 'SR', 'D', '200'),
    ('C', 'SR', 'D', '300'),
    ('C', 'SR', 'D', '400'),
    ('C', 'SR', 'D', '500'),
    ('C', 'SR', 'D', '750'),
    ('C', 'SR', 'D', '1000'),
    ('C', 'SR', 'D', '1250'),
    ('C', 'SR', 'D', '1500'),
    ('C', 'SR', 'D', '1750'),
    ('C', 'SR', 'D', '2048'),
    
    
        

    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'XR', 'D', '12'),
    ('C', 'XR', 'D', '50'),
    ('C', 'XR', 'D', '100'),
    ('C', 'XR', 'D', '200'),
    ('C', 'XR', 'D', '300'),
    ('C', 'XR', 'D', '400'),
    ('C', 'XR', 'D', '500'),
    ('C', 'XR', 'D', '750'),
    ('C', 'XR', 'D', '1000'),
    ('C', 'XR', 'D', '1250'),
    ('C', 'XR', 'D', '1500'),
    ('C', 'XR', 'D', '1750'),
    ('C', 'XR', 'D', '2048'),
    
    #12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    ('C', 'LR', 'W', '12'),
    ('C', 'LR', 'W', '14'),
    ('C', 'LR', 'W', '16'),
    ('C', 'LR', 'W', '18'),
    ('C', 'LR', 'W', '20'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'RR', 'W', '12'),
    ('C', 'RR', 'W', '50'),
    ('C', 'RR', 'W', '100'),
    ('C', 'RR', 'W', '200'),
    ('C', 'RR', 'W', '300'),
    ('C', 'RR', 'W', '400'),
    ('C', 'RR', 'W', '500'),
    ('C', 'RR', 'W', '750'),
    ('C', 'RR', 'W', '1000'),
    ('C', 'RR', 'W', '1250'),
    ('C', 'RR', 'W', '1500'),
    ('C', 'RR', 'W', '1750'),
    ('C', 'RR', 'W', '2048'),

    
    #12, 20,30,40,50,60,70,80,90, 100, 110, 130, 150
    ('C', 'SR', 'W', '12'),
    ('C', 'SR', 'W', '20'),
    ('C', 'SR', 'W', '30'),
    ('C', 'SR', 'W', '40'),
    ('C', 'SR', 'W', '50'),
    ('C', 'SR', 'W', '60'),
    ('C', 'SR', 'W', '70'),
    ('C', 'SR', 'W', '80'),
    ('C', 'SR', 'W', '90'),
    ('C', 'SR', 'W', '100'),
    ('C', 'SR', 'W', '110'), #ERROR
    ('C', 'SR', 'W', '130'),
    ('C', 'SR', 'W', '150'),
    ('C', 'SR', 'W', '250'),
    ('C', 'SR', 'W', '350'),
    ('C', 'SR', 'W', '500'),
    ('C', 'SR', 'W', '700'),
    ('C', 'SR', 'W', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('C', 'XR', 'W', '12'),
    ('C', 'XR', 'W', '50'),
    ('C', 'XR', 'W', '100'),
    ('C', 'XR', 'W', '200'),
    ('C', 'XR', 'W', '300'),
    ('C', 'XR', 'W', '400'),
    ('C', 'XR', 'W', '500'),
    ('C', 'XR', 'W', '750'),
    ('C', 'XR', 'W', '1000'),
    ('C', 'XR', 'W', '1250'),
    ('C', 'XR', 'W', '1500'),
    ('C', 'XR', 'W', '1750'),
    ('C', 'XR', 'W', '2048'),
    
    #12,16,20,26,30,40,50,60,70,80,90,100
    ('G', 'LR', 'D', '12'),
    ('G', 'LR', 'D', '16'), #ERROR
    ('G', 'LR', 'D', '20'),
    ('G', 'LR', 'D', '26'), #ERROR
    ('G', 'LR', 'D', '30'),
    ('G', 'LR', 'D', '40'),
    ('G', 'LR', 'D', '50'),
    ('G', 'LR', 'D', '60'),
    ('G', 'LR', 'D', '70'),
    ('G', 'LR', 'D', '80'),
    ('G', 'LR', 'D', '90'),
    ('G', 'LR', 'D', '100'),
    ('G', 'LR', 'D', '120'),
    ('G', 'LR', 'D', '140'),
    ('G', 'LR', 'D', '160'),
    ('G', 'LR', 'D', '180'),
    ('G', 'LR', 'D', '260'),

    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'RR', 'D', '12'),
    ('G', 'RR', 'D', '50'),
    ('G', 'RR', 'D', '100'),
    ('G', 'RR', 'D', '200'),
    ('G', 'RR', 'D', '300'),
    ('G', 'RR', 'D', '400'),
    ('G', 'RR', 'D', '500'),
    ('G', 'RR', 'D', '750'),
    ('G', 'RR', 'D', '1000'),
    ('G', 'RR', 'D', '1250'),
    ('G', 'RR', 'D', '1500'),
    ('G', 'RR', 'D', '1750'),
    ('G', 'RR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'SR', 'D', '12'),
    ('G', 'SR', 'D', '50'),
    ('G', 'SR', 'D', '100'),
    ('G', 'SR', 'D', '200'),
    ('G', 'SR', 'D', '300'),
    ('G', 'SR', 'D', '400'),
    ('G', 'SR', 'D', '500'),
    ('G', 'SR', 'D', '750'),
    ('G', 'SR', 'D', '1000'),
    ('G', 'SR', 'D', '1250'),
    ('G', 'SR', 'D', '1500'),
    ('G', 'SR', 'D', '1750'),
    ('G', 'SR', 'D', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'XR', 'D', '12'),
    ('G', 'XR', 'D', '50'),
    ('G', 'XR', 'D', '100'),
    ('G', 'XR', 'D', '200'),
    ('G', 'XR', 'D', '300'),
    ('G', 'XR', 'D', '400'),
    ('G', 'XR', 'D', '500'),
    ('G', 'XR', 'D', '750'),
    ('G', 'XR', 'D', '1000'),
    ('G', 'XR', 'D', '1250'),
    ('G', 'XR', 'D', '1500'),
    ('G', 'XR', 'D', '1750'),
    ('G', 'XR', 'D', '2048'),
    
    #12, 16, 20, 26, 30, 40, 50, 60, 70, 80
    ('G', 'LR', 'W', '12'),
    ('G', 'LR', 'W', '16'), #ERROR
    ('G', 'LR', 'W', '20'), 
    ('G', 'LR', 'W', '26'), #ERROR
    ('G', 'LR', 'W', '30'),
    ('G', 'LR', 'W', '40'),
    ('G', 'LR', 'W', '50'),
    ('G', 'LR', 'W', '60'),
    ('G', 'LR', 'W', '70'),
    ('G', 'LR', 'W', '80'),
    ('G', 'LR', 'W', '90'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'RR', 'W', '12'),
    ('G', 'RR', 'W', '50'),
    ('G', 'RR', 'W', '100'),
    ('G', 'RR', 'W', '200'),
    ('G', 'RR', 'W', '300'),
    ('G', 'RR', 'W', '400'),
    ('G', 'RR', 'W', '500'),
    ('G', 'RR', 'W', '750'),
    ('G', 'RR', 'W', '1000'),
    ('G', 'RR', 'W', '1250'),
    ('G', 'RR', 'W', '1500'),
    ('G', 'RR', 'W', '1750'),
    ('G', 'RR', 'W', '2048'),
    
    #12, 20,30,40,50,60,70,80,90, 100, 110, 130, 150
    ('G', 'SR', 'W', '12'),
    ('G', 'SR', 'W', '20'),
    ('G', 'SR', 'W', '30'),
    ('G', 'SR', 'W', '40'),
    ('G', 'SR', 'W', '50'),
    ('G', 'SR', 'W', '60'),
    ('G', 'SR', 'W', '70'),
    ('G', 'SR', 'W', '80'),
    ('G', 'SR', 'W', '90'),
    ('G', 'SR', 'W', '100'),
    ('G', 'SR', 'W', '110'), #ERROR
    ('G', 'SR', 'W', '130'),
    ('G', 'SR', 'W', '150'),
    ('G', 'SR', 'W', '250'),
    ('G', 'SR', 'W', '350'),
    ('G', 'SR', 'W', '500'),
    ('G', 'SR', 'W', '700'),
    ('G', 'SR', 'W', '2048'),
    
    #12, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2048
    ('G', 'XR', 'W', '12'),
    ('G', 'XR', 'W', '50'),
    ('G', 'XR', 'W', '100'),
    ('G', 'XR', 'W', '200'),
    ('G', 'XR', 'W', '300'),
    ('G', 'XR', 'W', '400'),
    ('G', 'XR', 'W', '500'),
    ('G', 'XR', 'W', '750'),
    ('G', 'XR', 'W', '1000'),
    ('G', 'XR', 'W', '1250'),
    ('G', 'XR', 'W', '1500'),
    ('G', 'XR', 'W', '1750'),
    ('G', 'XR', 'W', '2048')]




df = get_evaluation(eval_method=["Running_Time","RBO_Score_0.9"]).reset_index()
approx_keys = [f"{key[0]}_{key[2]}_{key[1]}_{key[3]}" for key in rbo_runs]
df_rbo = df[df["index"].isin(approx_keys)]


df_D = df_rbo[df_rbo["index"].str.contains("_D_")]

df_D_dict = dict()
df_D_dict_colors = dict()
exact_shap_values_D = {"LR":6.731802940368652, "RR":49.28979778289795,"SR":419.9395990371704,"XR":30.09837508201599}

df_W = df_rbo[df_rbo["index"].str.contains("_W_")]
df_test = df_W[df_W["index"].str.contains("G_W_SR")]
print(df_test)
df_test2 = df_W[df_W["index"].str.contains("K_W_SR")]
print(df_test2)
df_W_dict = dict()
df_W_dict_colors = dict()
exact_shap_values_W = {"LR":30.49867606163025, "RR":1666.557268857956,"SR":153204.29993009567,"XR":576.2753148078918}

approximations = ["C", "G", "K", "S"]
colors = {"C":"blue", "G":"green","K":"orange","S":"purple"}

model_names = {"LR":"Linear Regression", "RR":"Random Forest","SR":"SVM","XR":"XGBoost"}
shap_names = {"C":"Copula SHAP", "G":"Gaussian SHAP","K":"Kernel SHAP","S":"Sample SHAP"}



'''
models = ["LR"]
#D - LR
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["RBO_Score_0.9"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=2.310240, color='green', linestyle=':', linewidth=1, label="Gaussian SHAP - Sample Lower Bound", alpha=0.6)
plt.axvline(x=2.817665, color='blue', linestyle=':', linewidth=1, label="Copula SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
models = ["RR"]
#D - RR
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["RBO_Score_0.9"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=8.44816780090332, color='blue', linestyle='-', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=7.491661071777344, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=11.016456842422485, color='purple', linestyle=':', linewidth=1, label="Sample SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''


'''
models = ["SR"]
#D - SR
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["RBO_Score_0.9"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=35.218860 , color='blue', linestyle='-', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=38.0098, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''



'''
#D - XR
models = ["XR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_D[(df_D["index"].str.contains(f"{approximation}_")) & (df_D["index"].str.contains(f"_{model}_"))]
        df_D_dict[f"{approximation}_{model}"]=df
        df_D_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_D[model]
        shap_name = model_names[model]
        
for key in df_D_dict.keys():
    plt.scatter(df_D_dict[key]["Running_Time"], df_D_dict[key]["RBO_Score_0.9"], color=df_D_dict_colors[key], label=shap_names[key[0]])
    df = df_D_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_D_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Diabetes Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=7.779833, color='blue', linestyle='-', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=7.018606, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''


'''
#W - LR
models = ["LR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["RBO_Score_0.9"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=19.628270 , color='blue', linestyle=':', linewidth=1, label="Copula SHAP - Sample Lower Bound", alpha=0.6)
plt.axvline(x=7.858782, color='green', linestyle=':', linewidth=1, label="Gaussian SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#W - RR
models = ["RR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["RBO_Score_0.9"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_W_dict_colors[key], linestyle='-')
   
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=304.774652 , color='blue', linestyle='-', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=80.803144, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=128.962151, color='purple', linestyle=':', linewidth=1, label="Sample SHAP - Sample Lower Bound", alpha=0.6)
plt.axvline(x=82.998192, color='yellow', linestyle=':', linewidth=1, label="Kernel SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#W - SR - 1 
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_")) & (df_W["index"] != "K_W_SR_10")]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["RBO_Score_0.9"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Wine Dataset")
plt.axvline(x=2962.636975, color='yellow', linestyle=':', linewidth=1, label="Kernel SHAP - Sample Lower Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()'''

'''
#W - SR - 2 
models = ["SR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["RBO_Score_0.9"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Wine Dataset (with Exact Value)")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.legend(title="Approximation", loc="best")
plt.show()'''


#W - XR
models = ["XR"]
exact_shap_value = None
for index_model, model in enumerate(models):
    for index_approximation, approximation in enumerate(approximations):
        df = df_W[(df_W["index"].str.contains(f"{approximation}_")) & (df_W["index"].str.contains(f"_{model}_"))]
        df_W_dict[f"{approximation}_{model}"]=df
        df_W_dict_colors[f"{approximation}_{model}"]=colors[approximation]
        exact_shap_value = exact_shap_values_W[model]
        shap_name = model_names[model]
        
for key in df_W_dict.keys():
    plt.scatter(df_W_dict[key]["Running_Time"], df_W_dict[key]["RBO_Score_0.9"], color=df_W_dict_colors[key], label=shap_names[key[0]])
    df = df_W_dict[key].sort_values("Running_Time")
    plt.plot(df["Running_Time"], df["RBO_Score_0.9"], color=df_W_dict_colors[key], linestyle='-')
    
plt.xlabel("Running Time in seconds")
plt.ylabel("RBO Score")
plt.title(f"RBO Comparison (p=0.9) - {shap_name} Wine Dataset")
plt.axvline(x=exact_shap_value, color='red', linestyle='--', linewidth=1, label="Exact SHAP")
plt.axvline(x=304.774652 , color='blue', linestyle='-', linewidth=1, label="Copula SHAP - Sample Upper Bound", alpha=0.6)
plt.axvline(x=87.576517, color='green', linestyle='-', linewidth=1, label="Gaussian SHAP - Sample Upper Bound", alpha=0.6)
plt.legend(title="Approximation", loc="best")
plt.show()