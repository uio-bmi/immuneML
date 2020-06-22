library(magrittr)

plot_distributions = function(data,
                              x,
                              y = "value",
                              x_lab=x,
                              y_lab=y,
                              type = "quasirandom",
                              color = "NULL",
                              color_lab=color,
                              palette = NULL,
                              facet_rows = c(),
                              facet_columns = c(),
                              facet_type = "wrap", # choose from: "wrap" or "grid"
                              facet_scales = "free", # choose from: "fixed", "free", "free_x", "free_y"
                              facet_switch = "NULL",
                              nrow = "NULL",
                              height,
                              width,
                              result_path,
                              result_name) {
  params = as.list(match.call())
  params[[1]] = NULL
  saveRDS(params, file.path(result_path, paste0(result_name, ".rds")))

  if (facet_switch == "NULL") facet_switch = NULL
  if (color == "NULL") color = NULL
  if (!is.null(palette)) palette = rjson::fromJSON(palette)

  facet_columns = as.character(facet_columns)
  facet_rows = as.character(facet_rows)

  palette = ggexp::generate_palette_ggplot(data[, color, drop = TRUE], palette)

  if ("ScaleDiscrete" %in% class(palette)) {
    data[, color] = as.factor(data[, color, drop = TRUE])
  }

  plot = ggexp::plot_distributions(
    data = data,
    pairwise_annotation = NULL,
    x = x,
    y = y,
    color = color,
    pairwise_annotation_label = "p.adj.signif",
    pairwise_annotation_exclude = "ns",
    facet_columns = facet_columns,
    facet_rows = facet_rows,
    facet_type = facet_type,
    #facet_switch = facet_switch,
    facet_scales = facet_scales,
    type = type,
    nrow = nrow
  ) + palette


  ggplot2::ggsave(
    file.path(result_path, paste0(result_name, ".pdf")),
    plot,
    height = height,
    width = width,
    limitsize = FALSE
  )
}
