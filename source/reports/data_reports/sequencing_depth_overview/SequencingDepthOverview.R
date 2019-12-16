source(file.path(here::here(), "source", "visualization", "PaletteUtils.R"))

library(magrittr)
library(ggplot2)

plot_sequencing_depth_overview = function(data,
                x,
                color = NULL,
                facets,
                palette = NULL,
                nrow_distributions,
                nrow_scatterplot,
                height_distributions,
                height_scatterplot,
                width,
                result_path,
                result_name) {

  if (color == "NULL") color = NULL
  if (!is.null(palette)) palette = rjson::fromJSON(palette)
  facets = as.character(facets)

  palette = generate_ggplot_palette(data[, color, drop = TRUE], palette)

  if ("ScaleDiscrete" %in% class(palette)) {
    data[, color] = as.factor(data[, color, drop = TRUE])
  }

  annotation = compute_annotations(data, facets)

  distributions = plot_distributions(data, x, color, annotation, facets, nrow_distributions) + theme(legend.position = "none")

  wide_data = tidyr::pivot_wider(data, names_from = "feature")

  scatterplot = plot_scatterplot(wide_data, color, facets, nrow_scatterplot)

  patchwork::wrap_plots(
    list(distributions + palette, scatterplot + palette),
    heights = c(height_distributions, height_scatterplot),
    ncol = 1
  ) + patchwork::plot_layout(guides = 'collect')

  ggsave(
    filename = file.path(result_path, paste0(result_name, ".pdf")),
    height = height_distributions + height_scatterplot,
    width = width
  )
}

compute_annotations = function(data, facets) {
  annotation = data %>%
    dplyr::group_by(.dots = c("feature", "frame_type", facets)) %>%
    dplyr::summarize(total = sum(value),
                     average = mean(value))
  return(annotation)
}

plot_distributions = function(data,
                              x,
                              color,
                              annotation,
                              facets,
                              nrow) {
  distributions = (
    ggplot(data, aes_string(
      x = x, y = "value", color = color
    )) +
      ggbeeswarm::geom_quasirandom(
        method = "tukeyDense",
        alpha = 0.5,
        shape = 1,
        dodge.width = 1,
        position = position_dodge(width = 0.75)
      ) +
      geom_boxplot(
        alpha = 0,
        width = 0.3,
        outlier.size = 0,
        position = position_dodge(width = 1)
      ) +
      ggexp::theme_ggexp() +
      ggexp::get_palette() +
      geom_text(
        data = annotation,
        x = Inf,
        y = -Inf,
        inherit.aes = FALSE,
        aes(label = paste0("  Sum: ", total, " \n  Average: ", average)),
        hjust = 1,
        vjust = -0.4,
        size = 2
      ) +
      labs(y = NULL)
  ) %>%
    ggexp::plot_facets(
      facet_columns = c(facets, "feature", "frame_type"),
      facet_rows = c(),
      facet_type = "wrap",
      facet_switch = NULL,
      facet_scales = "free",
      nrow = nrow
    )

}

plot_scatterplot = function(data, color, facets, nrow) {
  (
    ggplot(
      data,
      aes_string(x = "`Total Reads`", y = "`Unique Clonotypes`", color = color)
    ) +
      geom_point() +
      ggexp::get_palette() +
      ggexp::theme_ggexp()
  ) %>%
    ggexp::plot_facets(
      facet_columns = c("frame_type", facets),
      facet_rows = c(),
      facet_type = "wrap",
      facet_switch = NULL,
      facet_scales = "free",
      nrow = nrow
    )
}
