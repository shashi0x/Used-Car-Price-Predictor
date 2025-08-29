# Getting data for our model

This is the first time i am working with a real world data where i will be first scrapping training data from a real life website like carwale.com and then clean it, turn it into a numpy array on which i can train my model.  
  
## Here's how i will be doing it

1. First we will analyse the structure of webpage that lists cars about how they get/store data.
2. luckily for carwale.com, i got the json list in the html code of webpage itself. But each page only have 25-35 listings.
3. to get more data we either need to add /page-2/ at the end of url or try for a different city. link: [https://carwale.com/used/delhi/](https://carwale.com/used/delhi/)
4. upon analysing, i found trying different cities to be easier.
5. So i got a list top 500 cities in india, and then iterated over that list, grabbing every listing from first page of these cities.
6. This way, i now have 8718 examples with different parameters and price.

### Here's example on how my Numpy array will look

|age|brand|kmDriven|fuelType|bodyType|seatCap|Owners|transmission|Price|
|---|-----|--------|--------|--------|-------|------|------------|----|
|2|12|20569|3|2|5|2|1|2000000|  

<!-- ### Here's what each number represent in brand:
 -->
