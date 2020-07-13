library(magrittr)
library(ggexp)
library(ggplot2)

plot_distribution = function(data,
                             x,
                             y,
                             color,
                             group,
                             type,
                             height,
                             width,
                             result_path,
                             result_name,
                             facet_rows = c(),
                             facet_columns = c(),
                             facet_type = "grid", # choose from: "wrap" or "grid"
                             facet_scales = "free", # choose from: "fixed", "free", "free_x", "free_y"
                             facet_switch = "NULL", # choose from: "null", "x", "y", "both"
                             nrow = "NULL",
                             ncol = "NULL",
                             x_lab = x,
                             y_lab = y,
                             color_lab = color,
                             palette = NULL) {
  params = as.list(match.call())
  params[[1]] = NULL
  saveRDS(params, file.path(result_path, paste0(result_name, ".rds")))

  for (var in formalArgs(plot_distribution)) {
    if (length(get(var)) == 1 &&
        toupper(get(var)) == "NULL")
      assign(var, NULL)
  }

  if (is.character(data)) {
      data = readr::read_csv(data)
  }

  if (!is.null(palette))
    palette = rjson::fromJSON(palette)

  facet_columns = as.character(facet_columns)
  facet_rows = as.character(facet_rows)

  palette = generate_palette_ggplot(data[, color, drop = TRUE], palette)

  if ("ScaleDiscrete" %in% class(palette)) {
    data[, color] = as.factor(data[, color, drop = TRUE])
  }

  plot = plot_distributions(
    data = data,
    x = x,
    y = y,
    color = color,
    group = group,
    facet_columns = facet_columns,
    facet_rows = facet_rows,
    facet_type = facet_type,
    facet_switch = facet_switch,
    facet_scales = facet_scales,
    type = type,
    nrow = nrow,
    ncol = ncol
  ) + palette + labs(x = x_lab, y = y_lab, fill = color_lab)

  ggsave(
    file.path(result_path, paste0(result_name, ".pdf")),
    plot,
    height = height,
    width = width,
    limitsize = FALSE
  )
}
