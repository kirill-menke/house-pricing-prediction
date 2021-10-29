# ('normalize', QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=0), 
# ["area_total", "area_kitchen", "area_living", "floor", "ceiling", "latitude", "longitude", "constructed", "stories"]),


# Mappings
X_train["seller"] = X_train["seller"].map({0.0:"Owner", 1.0:"Company", 2.0:"Agents", 3.0:"Developer"})
X_train["layout"] = X_train["layout"].map({0.0:"Adjacent", 1.0:"Isolated", 2.0:"Adjacent_isolated"})
X_train["condition"] = X_train["condition"].map({0.0:"Undecorated", 1.0:"Decorated", 2.0:"Euro_repair", 3.0:"Special_design"})
X_train["district"] = X_train["district"].map({0.0:"Central", 1.0:"North", 2.0:"North-East", 3.0:"East", 4.0:"South-East", 5.0:"South", 6.0:"South-West", 
7.0:"West", 8.0:"North-West", 9.0:"Zelenograd", 10.0:"Troitsk", 11.0:"Novomoskovsk"})
X_train["material"] = X_train["material"].map({0.0:"Bricks", 1.0:"Wood", 2.0:"Monolith", 3.0:"Panel", 4.0:"Block", 5.0:"Monolithic_brick", 6.0:"Stalin_project"})
X_train["parking"] = X_train["parking"].map({0.0:"Ground", 1.0:"Underground", 2.0:"Multilevel"})
X_train["heating"] = X_train["heating"].map({0.0:"Central", 1.0:"Individual", 2.0:"Boiler", 3.0:"Autonomous_boiler"})

X_val["seller"] = X_val["seller"].map({0.0:"Owner", 1.0:"Company", 2.0:"Agents", 3.0:"Developer"})
X_val["layout"] = X_val["layout"].map({0.0:"Adjacent", 1.0:"Isolated", 2.0:"Adjacent_isolated"})
X_val["condition"] = X_val["condition"].map({0.0:"Undecorated", 1.0:"Decorated", 2.0:"Euro_repair", 3.0:"Special_design"})
X_val["district"] = X_val["district"].map({0.0:"Central", 1.0:"North", 2.0:"North-East", 3.0:"East", 4.0:"South-East", 5.0:"South", 6.0:"South-West", 
7.0:"West", 8.0:"North-West", 9.0:"Zelenograd", 10.0:"Troitsk", 11.0:"Novomoskovsk"})
X_val["material"] = X_val["material"].map({0.0:"Bricks", 1.0:"Wood", 2.0:"Monolith", 3.0:"Panel", 4.0:"Block", 5.0:"Monolithic_brick", 6.0:"Stalin_project"})
X_val["parking"] = X_val["parking"].map({0.0:"Ground", 1.0:"Underground", 2.0:"Multilevel"})
X_val["heating"] = X_val["heating"].map({0.0:"Central", 1.0:"Individual", 2.0:"Boiler", 3.0:"Autonomous_boiler"})

print(X_train[["seller", "layout", "condition", "district", "material", "parking", "heating"]].dtypes)



# Try to utilize the automatic encoding auf catboost
X_train[categorical_columns] = X_train[categorical_columns].astype(str)
X_val[categorical_columns] = X_val[categorical_columns].astype(str)