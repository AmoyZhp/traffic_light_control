import trainer

if __name__ == "__main__":
    id_ = "independent"
    tr = trainer.get_trainer(id_, {})
    tr.run()
