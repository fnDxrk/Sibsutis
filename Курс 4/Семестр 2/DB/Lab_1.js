/*
  * 8.Найти все BMW с мощностью более 250 л.с., произведенные после 2021 года. 
  * Отобразить только поля brand, model, cost, power, production_date, исключить поле _id из вывода
*/

db.collection.find(
  {
    brand: "BMW",
    power: { $gt: 250 },
    production_date: { $gt: new Date("2022-01-01") }
  },
  {
    brand: 1,
    model: 1,
    cost: 1,
    power: 1,
    production_date: 1,
    _id: 0
  }
)

/* 
  * 9.Найти пять автомобилей: либо с оценкой больше 4.5 и (панорамной крышей или проекционным дисплеем),
  * либо без санкций и стоимостью до 6 млн.
*/

db.collection.find(
  {
    $or: [
      {
        $and: [
          { auction_evaluation: { $gt: 4.5 }},
          {
            $or: [
              { equipment: "панорамная крыша" },
              { equipment: "проекционный дисплей" }
            ]
          }
        ]
      },
      {
        $and: [
          { sanctions: false },
          { cost: { $lt: 6000000 } }
        ]
      }
    ]
  },
  {
    brand: 1,
    model: 1,
    cost: 1,
    auction_evaluation: 1,
    sanctions: 1,
    equipment: 1,
    _id: 0
  }
).limit(5)

/* 
  * 10.	Найти все автомобили владельцев с email в домене gmail.com,
  * у которых есть поле "equipment" хотя бы с 5 элементами. 
  * Отсортировать по убыванию стоимости автомобилей.
*/
db.collection.find(
  {
    "owner.email": /gmail\.com$/,
    $expr: { $gte: [{ $size: "$equipment" }, 5] }
  },
  {
    brand: 1,
    model: 1,
    cost: 1,
    "owner.email": 1,
    equipment: 1,
    _id: 0
  }
).sort({ cost: -1 })
