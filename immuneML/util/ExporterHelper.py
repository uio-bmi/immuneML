class ExporterHelper:

    @staticmethod
    def export_dataset(dataset, exporters, result_path):
        dataset_name = dataset.name if dataset.name is not None else dataset.identifier
        paths = {dataset_name: {}}
        formats = []

        if exporters is not None and len(exporters) > 0:
            for exporter in exporters:
                export_format = exporter.__name__[:-8]
                path = result_path / f"exported_dataset/{exporter.__name__.replace('Exporter', '').lower()}/"
                exporter.export(dataset, path)
                paths[dataset_name][export_format] = path
                formats.append(export_format)

        return {"paths": paths, "formats": formats}