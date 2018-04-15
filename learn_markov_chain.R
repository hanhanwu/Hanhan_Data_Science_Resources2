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


path<- "[change to your own file folder path]"  # change to your own file folder path
setwd(path)

# download the data from: https://www.kaggle.com/c/instacart-market-basket-analysis/data
# load data
orders <- fread("orders.csv", select = c("order_id", "user_id", "order_number", "eval_set"))
orders_prior <- fread("order_products__prior.csv", drop="add_to_cart_order")
head(orders)
head(orders_prior)

# group products in each transaction (order)
## group by order_id, you can also group by user_id
product_lst <- orders_prior[, .(current_order = list(product_id)), by=order_id]
head(product_lst)

# join product_lst with orders through order_id
order_lst <- merge(orders, product_lst, by="order_id")
head(order_lst)

# append previous order to each order, like lag function
## NOTE: sometimes this may show error but return the right output
setorderv(order_lst, c("user_id", "order_number"))  # reorder by reference, faster
head(order_lst)
order_lst[, previous_order:= shift(list(current_order)), by=user_id]  # shift for each user_id
head(order_lst)

# define set operations & generate transitions
intersect <- function(x,y) y[match(x,y,0L)]  # generates item set that both in x,y sets
setdiff <- function(x,y) x[match(x,y,0L) == 0L]  # generate item set that in x but not in y
## T11 means in both previous_order and current_order; T10 means in previous_order but not in current_order, etc.
order_lst[order_number>1, T11 := mapply(intersect, previous_order, current_order)] # only apply to order_number>1 for each user_id
order_lst[order_number>1, T10 := mapply(setdiff, previous_order, current_order)] # in previous order but not in current order
order_lst[order_number>1, T01 := mapply(setdiff, current_order, previous_order)]
head(order_lst)

# count products, each bin represent a product_id, the value of the bin records the count
max_product_id <- max(orders_prior$product_id) # 49688
countTransition <- function(L) tabulate(unlist(L), nbins = max_product_id)

# total transitions out of a state
## it's easy to get confused here, you can imagine each unique transition has a bin which records the count of this transition
order_lst[, n_orders := max(order_number), by=user_id]
head(order_lst)
N <- order_lst[, sum(n_orders-1)] # it won't transit out of the last state
N

# transitions out of state 1
N1 <- order_lst[order_number>1, countTransition(previous_order)]
head(N1)
length(N1)
N11 <- order_lst[order_number>1, countTransition(T11)]
head(N11)
N10 <- order_lst[order_number>1, countTransition(T10)]
head(N10)

# transitions our if state 0
N0 <- N - N1
head(N0)
N01 <- order_lst[order_number>1, countTransition(T01)]
N00 <- N0 - N01
head(N00)
head(N01)

# probability of state transition
## I don't quite understand +1, +2 here
state_transit_prob <- data.table(
  product_id = 1:max_product_id,
  # transition probability out of state 0
  P0 = (N0+1)/(N+2),
  P00 = (N00+1)/(N0+2),
  P01 = (N01+1)/(N0+2),
  # transition probability out of state 1
  P1 = (N1+1)/(N+2),
  P10 = (N10+1)/(N1+2),
  P11 = (N11+1)/(N1+2)
)
head(state_transit_prob)
