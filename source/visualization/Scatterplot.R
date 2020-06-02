library(ggplot2)

plot_two_dataframes <- function(df1, df2, label1, label2, x_label, y_label, result_path, result_name){
  #' plots two dataframes as scatter plots, where the dataframes have columns x and y for corresponding axes

  plot <- ggplot(df1, aes(x=x, y=y)) + geom_point(aes(color=label1), alpha=0.5) +
    geom_point(data=df2, mapping = aes(x=x, y=y, color=label2), alpha=0.5) +
    theme(axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 10, l = 0)),
          axis.title.y = element_text(margin = margin(t = 0, r = 10, b = 0, l = 10)),
          legend.key = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = 'white', colour = 'white'),
          panel.grid.minor = element_line(color="lightgrey", size=0.1), panel.grid.major = element_line(color="lightgrey", size=0.1),
          axis.line = element_line(color="gray57")
          ) +
    scale_colour_manual("", values = c("darkolivegreen3", 'cornflowerblue')) +
    xlab(x_label) + ylab(y_label)

  ggplot2::ggsave(
    file.path(result_path, paste0(result_name, ".pdf")),
    plot,
    height = 6,
    width = 8,
    limitsize = FALSE,
    units="in"
  )
}

