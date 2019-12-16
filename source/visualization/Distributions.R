source(file.path(here::here(), "source", "visualization", "PaletteUtils.R"))

library(magrittr)

plot_distributions = function(data,
                              x,
                              type = "quasirandom",
                              color = NULL,
                              group = NULL,
                              palette = NULL,
                              facet_rows,
                              facet_columns,
                              facet_type,
                              facet_scales,
                              facet_switch,
                              nrow,
                              height,
                              width,
                              result_path,
                              result_name) {
  params = as.list(match.call())
  params[[1]] = NULL
  saveRDS(params, file.path(result_path, paste0(result_name, ".rds")))

  if (facet_switch == "null") facet_switch = NULL
  if (color == "NULL") color = NULL
  if (group == "NULL") group = NULL
  if (!is.null(palette)) palette = rjson::fromJSON(palette)

  facet_columns = as.character(facet_columns)
  facet_rows = as.character(facet_rows)

  palette = generate_ggplot_palette(data[, color, drop = TRUE], palette)

  if ("ScaleDiscrete" %in% class(palette)) {
    data[, color] = as.factor(data[, color, drop = TRUE])
  }

  print(paste0(facet_type, facet_switch, facet_scales))

  plot = ggexp::plot_distributions(
    data = data,
    pairwise_annotation = NULL,
    x = x,
    y = "value",
    color = color,
    pairwise_annotation_label = "p.adj.signif",
    pairwise_annotation_exclude = "ns",
    facet_columns = facet_columns,
    facet_rows = facet_rows,
    facet_type = facet_type,
    facet_switch = facet_switch,
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
