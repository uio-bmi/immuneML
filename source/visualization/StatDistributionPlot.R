library(ggplot2)

plot_beta_distribution_binary_class <- function(alpha0, beta0, alpha1, beta1, x_label, label0, label1, upper_limit, lower_limit,
                                                result_path, result_name) {

  plot <- ggplot(data.frame(x = c(lower_limit, upper_limit)), aes(x = x)) + xlab(x_label) +
    theme(axis.line.y=element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          axis.title.y = element_blank(), axis.line.x = element_line(color="black"),
          axis.title.x = element_text(margin = margin(t = 20, r = 0, b = 10, l = 0)),
          legend.key = element_rect(fill = "transparent", color = NA),
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_rect(fill = 'white', colour = 'white')) +
    stat_function(fun = dbeta, args = list(shape1=alpha0, shape2=beta0), aes(color=label0)) +
    stat_function(fun = dbeta, args = list(shape1=alpha1, shape2=beta1), aes(color=label1)) +
    scale_colour_manual("", values = c("#E69F00", "#0072B2")) +
    scale_x_continuous(labels = function(x) format(x, scientific = FALSE))

  ggplot2::ggsave(
    file.path(result_path, paste0(result_name, ".pdf")),
    plot,
    height = 6,
    width = 8,
    limitsize = FALSE,
    units="in"
  )
}