library(ggplot2)
library(ggexp)

plot_barplot = function(data,
                        x,
                        y,
                        color,
                        height,
                        width,
                        result_path,
                        result_name,
                        errorbar_meaning = "se", # choose from: "se", "sd", "ci"
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
                        palette = NULL,
                        sort_by_y = FALSE,
                        ml_benchmark = FALSE) {

  params = as.list(match.call())
  params[[1]] = NULL
  saveRDS(params, file.path(result_path, paste0(result_name, ".rds")))

  for (var in formalArgs(plot_barplot)) {
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

  summary = Rmisc::summarySE(
    data,
    measurevar = y,
    groupvars = unique(c(facet_columns, facet_rows, x, color)),
    na.rm = TRUE
  )

  if (sort_by_y) {
    summary$plotting_x = reorder(summary[[x]],-abs(summary[[y]]))
  } else {
    summary$plotting_x = summary[[x]]
  }

  plot = ggplot(summary, aes_string(x = "plotting_x", y = y, fill = color)) +
    geom_bar(stat = "identity", position = "dodge") +
    geom_errorbar(
      aes(ymin = summary[[y]] - summary[[errorbar_meaning]], ymax = summary[[y]] +
            summary[[errorbar_meaning]]),
      size = 0.5,
      width = .25,
      position = position_dodge(.9)
    ) +
    theme_ggexp() + labs(x = x_lab, y = y_lab, fill = color_lab) +
    palette

  if (ml_benchmark) {
    plot = plot + theme(
      legend.position = "bottom",
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank()
    )
  }

  plot = plot_facets(plot,
                     facet_rows,
                     facet_columns,
                     facet_type,
                     facet_scales,
                     facet_switch,
                     nrow,
                     ncol)

  ggsave(
    file.path(result_path, paste0(result_name, ".pdf")),
    plot,
    height = height,
    width = width,
    limitsize = FALSE,
    units = "in"
  )
}
