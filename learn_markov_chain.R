library(DiagrammeR)
library(data.table)

# visualize a transition system
## The shape here controls the shape of each state
grViz("
      digraph markov_chain {
        graph[overlap=true, fontsize=10, rankdir=LR]
        node[shape=circle] 0;1;
        0->0 0->1 1->0 1->1
      }", height=125)


path<- "[change to your file folder path]"  # change to your file folder path
setwd(path)

# download the data from: https://www.kaggle.com/c/instacart-market-basket-analysis/data
# load data
orders <- fread("orders.csv", select = c("order_id", "user_id", "order_number", "eval_set"))
orders_prior <- fread("order_products__prior.csv", drop="add_to_cart_order")
head(orders)
head(orders_prior)

# group products in each transaction (order)
product_lst <- orders_prior[, .(current_order = list(product_id)), by=order_id]
head(product_lst)

# join product_lst with orders through order_id
order_lst <- merge(orders, product_lst, by="order_id")
head(order_lst)

# append previous order to each order, like lag function
## NOTE: this will show error but return the right output
setorderv(order_lst, c("user_id", "order_number"))  # reorder by reference, faster
head(order_lst)
order_lst[, previous_order:= shift(list(current_order)), by=user_id]
head(order_lst)
