using BoostingDecisionTrees: majority_label
using StatsBase: countmap
using Profile

# Erzeuge komplexere Daten: 100.000 Samples mit 10 möglichen Labels
#const LABELS = ["label_$i" for i in 1:10]
#y_large = rand(LABELS, 100_000)  # 100.000 Samples, 10 mögliche Labels

y_large = rand(1:10, 1_000_000)

function profile_majority_label(n)
    for i = 1:n
        majority_label(y_large)
    end
end

#@profview profile_majority_label(10)
@profile profile_majority_label(1000)
Profile.print(format=:flat, sortedby=:time)