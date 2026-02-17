// Task 1
db.sales.aggregate([
  {
    $match: {
      production_date: {
        $gte: ISODate("2020-01-01"),
        $lt: ISODate("2022-01-01")
      }
    }
  },
  {
    $group: {
      _id: {
        brand: "$brand",
        year: { $year: "$production_date" }
      },
      cars_sold: { $sum: 1 },
      total_cost: { $sum: "$cost" }
    }
  },
  {
    $match: {
      cars_sold: { $gte: 2 }
    }
  },
  {
    $project: {
      _id: 0,
      title: {
        $concat: ["$_id.brand", " ", { $toString: "$_id.year" }]
      },
      total_cost: 1,
      cars_sold: 1,
      popularity: {
        $switch: {
          branches: [
            {
              case: { $gt: ["$cars_sold", 6] },
              then: "выбор миллионов"
            },
            {
              case: {
                $and: [
                  { $gte: ["$cars_sold", 4] },
                  { $lte: ["$cars_sold", 6] }
                ]
              },
              then: "популярный"
            }
          ],
          default: "нишевый"
        }
      }
    }
  },
  {
    $out: "stats"
  }
])

// Task 2
db.sales.aggregate([
  { $unwind: "$equipment" },
  {
    $group: {
      _id: "$brand",
      options: { $addToSet: "$equipment" }
    }
  },
  {
    $project: {
      _id: 0,
      brand: "$_id",
      options: 1
    }
  },
  {
    $out: "brands"
  }
])

// Task 3
db.sales.aggregate([
  {
    $match: {
      owner: { $type: "string" }
    }
  },
  {
    $group: {
      _id: "$owner",
      avg_auction_evaluation: { $avg: "$auction_evaluation" }
    }
  },
  {
    $lookup: {
      from: "dealers",
      localField: "_id",
      foreignField: "name",
      as: "dealer"
    }
  },
  { $unwind: "$dealer" },
  {
    $project: {
      _id: "$dealer._id",
      avg_auction_evaluation: { $round: ["$avg_auction_evaluation", 2] }
    }
  },
  {
    $merge: {
      into: "dealers",
      whenMatched: "merge",
      whenNotMatched: "discard"
    }
  }
])

// Task 4
db.sales.aggregate([
  {
    $match: {
      owner: { $type: "string" }
    }
  },
  {
    $group: {
      _id: "$owner",
      total: { $sum: 1 },
      sanctioned: {
        $sum: {
          $cond: [{ $eq: ["$sanctions", true] }, 1, 0]
        }
      }
    }
  },
  {
    $addFields: {
      sanctions_share: {
        $cond: [
          { $gt: ["$total", 0] },
          { $divide: ["$sanctioned", "$total"] },
          0
        ]
      }
    }
  },
  {
    $match: {
      sanctions_share: { $gt: 0.5 }
    }
  },
  {
    $lookup: {
      from: "dealers",
      localField: "_id",
      foreignField: "name",
      as: "dealer"
    }
  },
  { $unwind: "$dealer" },
  {
    $project: {
      _id: 0,
      name: "$dealer.name",
      address: "$dealer.address"
    }
  }
])
